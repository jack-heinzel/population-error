from functools import partial
import numpy as np
import jax.numpy as jnp
import jax
import jax_tqdm
import bilby
import gwpopulation
import pandas as pd

def selection_function(weights, total_generated):
    """
    Compute the selection function given weights and total number of injections.

    Parameters
    ----------
    weights : jnp.ndarray
        Array of importance weights for injection samples.
    total_generated : int or float
        Total number of injections.

    Returns
    -------
    float
        Estimated selection function value (mean weight normalized by total samples).
    """
    return jnp.sum(weights) / total_generated

def selection_function_log_covariance(weights_n, weights_m, total_generated):
    """
    Compute the covariance of log selection functions between two weight sets.

    Parameters
    ----------
    weights_n : jnp.ndarray
        First set of importance weights.
    weights_m : jnp.ndarray
        Second set of importance weights (must match shape of weights_n).
    total_generated : int or float
        Total number of injections.

    Returns
    -------
    float
        Covariance between log selection function estimates.
    """
    assert weights_n.shape == weights_m.shape
    mu_n, mu_m = selection_function(weights_n, total_generated), selection_function(weights_m, total_generated)
    cov = jnp.sum(weights_n * weights_m) / total_generated / mu_n / mu_m - 1
    return cov / (total_generated-1)

def likelihood_log_correction(weights, total_generated, Nobs):
    """
    Compute the likelihood log-correction term for variance estimation.

    Parameters
    ----------
    weights : jnp.ndarray
        Importance weights for injection samples.
    total_generated : int or float
        Total number of injections.
    Nobs : int
        Number of observed events.

    Returns
    -------
    float
        Likelihood log-correction value.
    """
    var = selection_function_log_covariance(weights, weights, total_generated)
    return Nobs * (Nobs+1) * var / 2

def reweighted_event_bayes_factors(event_pe_weights, counts=None):
    """
    Compute reweighted Bayes factors for a set of events.

    Parameters
    ----------
    event_pe_weights : jnp.ndarray
        Array of shape (Nobs, NPE) with posterior sample weights per event.
    counts : jnp.ndarray, optional
        Per-event real sample counts, shape (Nobs,). Provide when ``event_pe_weights``
        is zero-padded (ragged) so the per-event mean divides by the true count rather
        than ``NPE``. If ``None``, a plain mean over axis 1 is used.

    Returns
    -------
    jnp.ndarray
        Array of mean Bayes factors per event, shape (Nobs,).
    """

    if counts is None:
        return jnp.mean(event_pe_weights, axis=1)
    return jnp.sum(event_pe_weights, axis=1) / counts

def event_log_covariances(event_pe_weights_n, event_pe_weights_m, counts=None):
    """
    Compute covariances of log Bayes factors between two sets of event weights.

    Parameters
    ----------
    event_pe_weights_n : jnp.ndarray
        First array of event posterior sample weights, shape (Nobs, NPE).
    event_pe_weights_m : jnp.ndarray
        Second array of event posterior sample weights (same shape as above).
    counts : jnp.ndarray, optional
        Per-event real sample counts, shape (Nobs,), for zero-padded (ragged) weights.
        If ``None``, the common ``NPE`` is used for all events.

    Returns
    -------
    jnp.ndarray
        Covariances per event, shape (Nobs,).
    """

    assert event_pe_weights_m.shape == event_pe_weights_n.shape
    Nobs, NPE = event_pe_weights_n.shape

    mu_n = reweighted_event_bayes_factors(event_pe_weights_n, counts=counts)
    mu_m = reweighted_event_bayes_factors(event_pe_weights_m, counts=counts)

    if counts is None:
        cov = jnp.mean(event_pe_weights_n*event_pe_weights_m, axis=1) / mu_n / mu_m - 1
        return cov / (NPE - 1)
    cov = (jnp.sum(event_pe_weights_n*event_pe_weights_m, axis=1) / counts) / mu_n / mu_m - 1
    return cov / (counts - 1)

def log_likelihood_covariance(vt_weights_n, vt_weights_m, event_pe_weights_n, event_pe_weights_m, total_generated, event_counts=None):
    """
    Compute covariance of log-likelihood estimates from injection and event weights.

    Parameters
    ----------
    vt_weights_n : jnp.ndarray
        Injection weights for the first hyperposterior sample.
    vt_weights_m : jnp.ndarray
        Injection weights for the second hyperposterior sample.
    event_pe_weights_n : jnp.ndarray
        Event posterior weights for the first sample, shape (Nobs, NPE).
    event_pe_weights_m : jnp.ndarray
        Event posterior weights for the second sample, shape (Nobs, NPE).
    total_generated : int or float
        Total number of injections.

    Returns
    -------
    float
        Log-likelihood covariance estimate.
    """

    Nobs, NPE = event_pe_weights_n.shape

    event_covs = event_log_covariances(event_pe_weights_n, event_pe_weights_m, counts=event_counts)
    vt_cov = selection_function_log_covariance(vt_weights_n, vt_weights_m, total_generated)

    return jnp.sum(event_covs) + Nobs**2 * vt_cov

def error_statistics_from_weights(vt_weights, event_weights, total_generated, include_likelihood_correction=True, event_counts=None):
    """
    Compute error statistics for hyperposterior, Eqs. 36-39 of arxiv:2509.07221

    Parameters
    ----------
    vt_weights : jnp.ndarray
        Array of shape (n_samples, n_injections), injection weights per hyperposterior sample.
    event_weights : jnp.ndarray
        Array of shape (n_samples, n_obs, n_pe), event posterior weights per hyperposterior sample.
    total_generated : int or float
        Total number of injections.
    include_likelihood_correction : bool, default=True
        Whether to include the likelihood correction term in the accuracy statistic. Set to True if
        inference did not include the likelihood correction term, set to False if inference did
        include the likelihood correction.
    event_counts : jnp.ndarray, optional
        Per-event real sample counts, shape (n_obs,), when ``event_weights`` is
        zero-padded (ragged). Padded entries must be 0 (e.g. from a +inf sampling
        prior). If ``None``, all events are assumed to have the common ``n_pe``.

    Returns
    -------
    tuple of floats
        (precision, accuracy, error), where:
        - precision : float
            Expected information lost to uncertainty in posterior estimator.
        - accuracy : float
            Expected information lost to bias in posterior estimator
        - error : float
            Expected information lost to both bias and uncertainty in posterior estimator.
    """

    variances, weights, corrections = _covariance_terms_from_weights(
        vt_weights, event_weights, total_generated, include_likelihood_correction=include_likelihood_correction, event_counts=event_counts
        )

    # the likelihood correction shifts the bias only, so it enters the accuracy but not the precision
    precision = float((jnp.mean(variances) - jnp.mean(weights)) / 2 / jnp.log(2))
    accuracy = float(jnp.var(weights - corrections) / 2 / jnp.log(2))
    error = float(precision + accuracy)

    return {'error_statistic': error, 'precision_statistic': precision, 'accuracy_statistic': accuracy}

def _covariance_terms_from_weights(vt_weights, event_weights, total_generated, include_likelihood_correction=True, event_counts=None):
    """
    Per-hyperposterior-sample covariance terms used by the error statistics.

    Returns
    -------
    tuple of jnp.ndarray, each of shape (n_samples,)
        - variances : Var[ln L(Lambda_n)]
        - weights : mean over m of Cov[ln L(Lambda_n), ln L(Lambda_m)]
        - corrections : likelihood log-correction at Lambda_n (zeros if not included)
    """

    length, Nobs, NPE = event_weights.shape
    axis = jnp.arange(length)
    arr_n, arr_m = jnp.meshgrid(axis, axis, indexing='ij')
    f = lambda n, m: log_likelihood_covariance(vt_weights[n], vt_weights[m], event_weights[n], event_weights[m], total_generated, event_counts=event_counts)
    _f = lambda x: f(x, x)
    variances = jax.lax.map(_f, axis)

    @jax_tqdm.scan_tqdm(length, print_rate=1, tqdm_type='std')
    def weight_func(carry, n):
        _f = lambda x: f(arr_n[n,x], arr_m[n,x])
        meanw = jnp.mean(jax.lax.map(_f, axis), axis=0)
        if include_likelihood_correction:
            correction = likelihood_log_correction(vt_weights[n], total_generated, Nobs)
        else:
            correction = 0.
        return carry, (meanw, correction)

    _, (weights, corrections) = jax.lax.scan(weight_func, 0., xs=axis)
    return variances, weights, corrections

def _distinct_samples(hyperposterior, keys=None, seed=12345):
    """
    Identify repeated hyperposterior samples (e.g. from resampling a nested-sampling run).

    Rows are grouped by a random linear projection of their standardized values, which avoids
    forming the (n, n_params) array when there are very many hyperparameters, and every group is
    then checked to consist of identical rows. NaN is treated as equal to NaN.

    Returns
    -------
    keys : list of str
    representative : np.ndarray, shape (U,), index of one copy of each distinct sample
    multiplicity : np.ndarray, shape (U,), number of copies of each distinct sample
    """
    if keys is None:
        keys = list(hyperposterior.keys())
    rng = np.random.default_rng(seed)
    n = len(np.asarray(hyperposterior[keys[0]]))
    projection = np.zeros(n)
    for key in keys:
        column = np.asarray(hyperposterior[key], dtype=float)
        finite = np.isfinite(column)
        standardized = np.zeros(n)
        if finite.any():
            scale = column[finite].std()
            standardized[finite] = (column[finite] - column[finite].mean()) / (scale if scale > 0 else 1.)
        # non-finite values get fixed random values, so identical rows still project identically
        standardized[np.isnan(column)] = rng.normal()
        standardized[column == np.inf] = rng.normal()
        standardized[column == -np.inf] = rng.normal()
        projection += standardized * rng.normal()
    _, representative, inverse, multiplicity = np.unique(projection, return_index=True, return_inverse=True, return_counts=True)

    copies = np.flatnonzero(multiplicity[inverse] > 1)
    for key in keys:
        column = np.asarray(hyperposterior[key], dtype=float)
        a, b = column[copies], column[representative[inverse[copies]]]
        if not np.all((a == b) | (np.isnan(a) & np.isnan(b))):
            # projections of distinct rows collided; fall back to an exact comparison of all columns
            theta = np.column_stack([np.asarray(hyperposterior[k], dtype=float) for k in keys])
            theta = np.where(np.isnan(theta), np.inf, theta)
            _, representative, multiplicity = np.unique(theta, axis=0, return_index=True, return_counts=True)
            break
    return keys, representative, multiplicity

def _invalid_marginal_parameter(x, key):
    """
    Return True, warning if needed, for a hyperparameter whose marginal statistics are undefined:
    one that takes a single value, or has non-finite values.
    """
    if not np.all(np.isfinite(x)):
        import warnings
        warnings.warn(f'Hyperparameter {key} has non-finite values; its marginal statistics are set to NaN.')
        return True
    return np.ptp(x) == 0

def _copy_groups(event_vars, rtol=1e-9):
    """
    Group distinct hyperposterior samples that are copies of the same population hyperparameters.

    Resampling a nested-sampling run repeats samples, and post-processing can then attach a
    different value of a derived quantity (e.g. a merger rate drawn from p(R | Lambda)) to each
    copy, so copies need not be identical rows. Copies share the single-event Monte Carlo
    integrals, which depend on the population shape but not on the rate, so they are grouped by
    their per-event log-variances. Distinct samples from a continuous hyperposterior have
    distinct values with probability one.

    Parameters
    ----------
    event_vars : np.ndarray, shape (U, Nobs)
        Per-event variance of the log single-event Monte Carlo integral, for each distinct sample.

    Returns
    -------
    np.ndarray, shape (U,), the copy group of each distinct sample.
    """
    event_vars = np.asarray(event_vars).reshape(len(event_vars), -1)
    order = np.lexsort(event_vars.T[::-1])
    sorted_vars = event_vars[order]
    new_group = np.concatenate([[True], ~np.all(np.isclose(sorted_vars[1:], sorted_vars[:-1], rtol=rtol, atol=0), axis=1)])
    groups = np.empty(len(order), dtype=int)
    groups[order] = np.cumsum(new_group) - 1
    return groups

def _window_half_width(groups, k_neighbours):
    """
    Half-width of the window of distinct samples (in sorted order) that contains the K nearest
    valid neighbours: at most (largest copy group - 1) samples in it are excluded as copies.
    """
    return int(k_neighbours + np.bincount(groups).max() - 1)

def _sorted_window_neighbours(xs, cs, gs, k_neighbours, half_width):
    """
    JAX kernel for :func:`_marginal_nearest_neighbours`, given values xs already sorted ascending,
    their multiplicities cs and copy groups gs. Returns (offsets, neighbour_weights), each of
    shape (U, 2 half_width).
    """
    U = xs.shape[0]
    window = jnp.concatenate([-jnp.arange(half_width, 0, -1), jnp.arange(1, half_width + 1)])
    positions = jnp.arange(U)[:, None] + window[None, :]
    clipped = jnp.clip(positions, 0, U - 1)
    valid = (positions >= 0) & (positions < U) & (gs[clipped] != gs[:, None])
    distance = jnp.where(valid, jnp.abs(xs[clipped] - xs[:, None]), jnp.inf)

    nearest = jnp.argsort(distance, axis=1, stable=True)
    offsets = jnp.take_along_axis(jnp.broadcast_to(window, positions.shape), nearest, axis=1)
    copies = jnp.where(jnp.take_along_axis(valid, nearest, axis=1), cs[jnp.take_along_axis(clipped, nearest, axis=1)], 0)
    # greedily take copies of the nearest samples until K have been used
    neighbour_weights = jnp.clip(k_neighbours - (jnp.cumsum(copies, axis=1) - copies), 0, copies)
    return offsets, neighbour_weights

def _marginal_nearest_neighbours(x, multiplicity, groups, k_neighbours, tiebreak):
    """
    K nearest neighbours in one hyperparameter, over distinct hyperposterior samples.

    Each distinct sample u is paired with the k_neighbours closest *other* samples in x,
    counting repeated samples with their multiplicity but never pairing a sample with a
    copy of itself, i.e. a sample in the same copy group (which would return Var rather than
    Cov). Only a window of distinct samples on either side in sorted order is searched (see
    :func:`_window_half_width`). Distinct samples with equal x are ordered randomly by tiebreak,
    so that neighbours within a tie are not systematically close in the other hyperparameters.

    Parameters
    ----------
    x : np.ndarray, shape (U,)
        Value of the hyperparameter for each distinct sample.
    multiplicity : np.ndarray, shape (U,)
        Number of copies of each distinct sample.
    groups : np.ndarray, shape (U,)
        Copy group of each distinct sample, from :func:`_copy_groups`.
    k_neighbours : int
        Number of nearest neighbours, K.
    tiebreak : np.ndarray, shape (U,)
        Independent random numbers, used to order samples with equal x.

    Returns
    -------
    order : np.ndarray, shape (U,)
        Distinct-sample index at each position of x sorted ascending.
    offsets : np.ndarray, shape (U, 2 half_width)
        Neighbour position minus own position in the sorted order.
    neighbour_weights : np.ndarray, shape (U, 2 half_width)
        Number of copies of each neighbour used; each row sums to K.
    """
    order = np.lexsort((tiebreak, x))
    offsets, neighbour_weights = _sorted_window_neighbours(
        jnp.asarray(x[order]), jnp.asarray(multiplicity[order]), jnp.asarray(groups[order]), k_neighbours, _window_half_width(groups, k_neighbours)
        )
    offsets, neighbour_weights = np.asarray(offsets), np.asarray(neighbour_weights)

    if np.any(neighbour_weights.sum(axis=1) < k_neighbours):
        raise ValueError(f'Fewer than k_neighbours={k_neighbours} distinct hyperposterior samples available.')
    return order, offsets, neighbour_weights

def _marginal_statistics(pair_covariance, b, multiplicity, order, offsets, neighbour_weights, n, joint_mean_cov):
    """
    Combine nearest-neighbour pair covariances into marginal (precision, accuracy, error).

    pair_covariance[p, j] is Cov[ln L] between the distinct samples at sorted positions p
    and p + offsets[p, j]; b is the per-distinct-sample bias term (weights - correction).
    """
    K = neighbour_weights.sum(axis=1)[0]
    outer = multiplicity[order][:, None] * neighbour_weights / n / K
    b_sorted = b[order]
    neighbour_b = b_sorted[np.clip(np.arange(len(order))[:, None] + offsets, 0, len(order) - 1)]
    b_mean = np.sum(multiplicity * b) / n

    precision = float((np.sum(outer * pair_covariance) - joint_mean_cov) / 2 / np.log(2))
    accuracy = float(np.sum(outer * (b_sorted[:, None] - b_mean) * (neighbour_b - b_mean)) / 2 / np.log(2))
    return {'error_statistic': precision + accuracy, 'precision_statistic': precision, 'accuracy_statistic': accuracy}

def marginal_error_statistics_from_weights(
        vt_weights,
        event_weights,
        total_generated,
        hyperposterior,
        parameters=None,
        k_neighbours=1,
        include_likelihood_correction=True,
        event_counts=None,
        verbose=True,
        ):
    """
    Compute error statistics for the one-dimensional marginal hyperposteriors.

    For Lambda = (Lambda', x), MC noise delta(Lambda) = ln L_hat - ln L perturbs the marginal
    posterior p(x) through its conditional average over p(Lambda' | x). The marginal precision
    replaces Var[delta(Lambda)] in the joint precision statistic by

        E_x Var[ E_{Lambda'|x} delta ] = E Cov[delta(Lambda', x), delta(Lambda'', x)],

    with Lambda', Lambda'' drawn independently from p(Lambda' | x). This is estimated by pairing
    each hyperposterior sample with its K nearest neighbours in x: the neighbours are chosen
    using x alone, so their Lambda' are independent draws from p(Lambda' | x ~ x_n). The
    marginal accuracy uses the same pairing to estimate Var_x of the conditional mean bias.
    Repeated hyperposterior samples are never paired with a copy of themselves, including copies
    that differ only in post-processed columns such as a merger rate drawn for each sample.

    Parameters
    ----------
    vt_weights : jnp.ndarray
        Array of shape (n_samples, n_injections), injection weights per hyperposterior sample.
    event_weights : jnp.ndarray
        Array of shape (n_samples, n_obs, n_pe), event posterior weights per hyperposterior sample.
    total_generated : int or float
        Total number of injections.
    hyperposterior : pandas.DataFrame or dict of array_like
        Hyperposterior samples, in the same order as the first axis of the weights. Repeated
        samples are identified automatically.
    parameters : list of str, optional
        Hyperparameters for which to compute marginal statistics. Defaults to all columns.
    k_neighbours : int, default=1
        Number of nearest neighbours in x averaged over for each sample. Larger K reduces the
        noise of the estimate at the cost of resolution in x.
    include_likelihood_correction : bool, default=True
        As in :func:`error_statistics_from_weights`.
    event_counts : jnp.ndarray, optional
        As in :func:`error_statistics_from_weights`.
    verbose : bool, default=True
        Whether to print a summary table.

    Returns
    -------
    dict
        - 'joint' : dict of the joint (error, precision, accuracy) statistics, as returned by
          :func:`error_statistics_from_weights`.
        - 'marginal' : dict mapping each parameter to its dict of (error, precision, accuracy)
          statistics. Parameters that take a single value or have non-finite values are mapped
          to NaNs. The estimates are noisy and can be slightly negative when the true value is ~0.
    """

    hyperposterior, n = format_hyperposterior(dict(hyperposterior) if isinstance(hyperposterior, dict) else hyperposterior)
    keys, representative, multiplicity = _distinct_samples(hyperposterior)
    tiebreak = np.random.default_rng(0).random(len(representative))
    if parameters is None:
        parameters = keys
    if n != vt_weights.shape[0] or n != event_weights.shape[0]:
        raise ValueError(f'hyperposterior has {n} samples but the weights have {vt_weights.shape[0]} and {event_weights.shape[0]}.')

    variances, weights, corrections = _covariance_terms_from_weights(
        vt_weights, event_weights, total_generated, include_likelihood_correction=include_likelihood_correction, event_counts=event_counts
        )
    variances, weights, corrections = np.asarray(variances), np.asarray(weights), np.asarray(corrections)
    event_vars = jax.lax.map(lambda w: event_log_covariances(w, w, counts=event_counts), event_weights[jnp.asarray(representative)])
    groups = _copy_groups(event_vars)
    joint_mean_cov = float(np.mean(weights))
    joint_precision = float((np.mean(variances) - joint_mean_cov) / 2 / np.log(2))
    joint_accuracy = float(np.var(weights - corrections) / 2 / np.log(2))
    joint = {'error_statistic': joint_precision + joint_accuracy, 'precision_statistic': joint_precision, 'accuracy_statistic': joint_accuracy}

    b = (weights - corrections)[representative]

    # gather the nearest-neighbour pairs for all parameters and evaluate their covariances together
    neighbours, pairs_n, pairs_m = {}, [], []
    for key in parameters:
        x = np.asarray(hyperposterior[key])[representative]
        if _invalid_marginal_parameter(x, key):
            continue
        order, offsets, neighbour_weights = _marginal_nearest_neighbours(x, multiplicity, groups, k_neighbours, tiebreak)
        neighbour_order = order[np.clip(np.arange(len(order))[:, None] + offsets, 0, len(order) - 1)]
        used = neighbour_weights > 0
        neighbours[key] = (order, offsets, neighbour_weights, used)
        pairs_n.append(representative[np.broadcast_to(order[:, None], used.shape)[used]])
        pairs_m.append(representative[neighbour_order[used]])

    if pairs_n:
        pairs_n, pairs_m = jnp.array(np.concatenate(pairs_n)), jnp.array(np.concatenate(pairs_m))
        pair_func = lambda nm: log_likelihood_covariance(
            vt_weights[nm[0]], vt_weights[nm[1]], event_weights[nm[0]], event_weights[nm[1]], total_generated, event_counts=event_counts
            )
        pair_covariances = np.asarray(jax.lax.map(pair_func, (pairs_n, pairs_m)))
        start = 0

    marginal = {}
    for key in parameters:
        if key not in neighbours:
            marginal[key] = {'error_statistic': np.nan, 'precision_statistic': np.nan, 'accuracy_statistic': np.nan}
            continue
        order, offsets, neighbour_weights, used = neighbours[key]
        pair_covariance = np.zeros(used.shape)
        pair_covariance[used] = pair_covariances[start:start + used.sum()]
        start += used.sum()
        marginal[key] = _marginal_statistics(pair_covariance, b, multiplicity, order, offsets, neighbour_weights, n, joint_mean_cov)

    if verbose:
        _print_marginal_statistics(joint, marginal)
    return {'joint': joint, 'marginal': marginal}

def _print_marginal_statistics(joint, marginal):
    width = max([len(k) for k in marginal] + [len('joint')])
    print(f"\n{'parameter':<{width}}  {'error':>10}  {'precision':>10}  {'accuracy':>10}  {'precision / joint':>17}")
    rows = [('joint', joint)] + list(marginal.items())
    for key, stats in rows:
        ratio = stats['precision_statistic'] / joint['precision_statistic'] if joint['precision_statistic'] != 0 else np.nan
        print(f"{key:<{width}}  {stats['error_statistic']:>10.4g}  {stats['precision_statistic']:>10.4g}  {stats['accuracy_statistic']:>10.4g}  {ratio:>17.3f}")
    print('(statistics in bits)')

def bilby_model_to_model_function(bilby_model, conversion_function=lambda args: (args, None), rate=False, rate_key='rate'):
    r"""
    Wrap a Bilby or gwpopulation jax-compatible model into a callable function interface.

    Note: if using the rate-full likelihood, this model should return dN/d\theta. It should
    *not* be in comoving rate density in units Gpc^{-3} yr^{-1}. Instead, it is expected to be in 
    a density in redshift.

    Parameters
    ----------
    bilby_model : bilby.hyper.model.Model or callable Model object to be converted. If it is already 
        a callable, it is returned unchanged.
    conversion_function : callable, optional
        Function applied to parameter dictionaries before evaluating the model.
        Should take a dict of parameters and return (modified_parameters, added_keys).
    rate : bool, default=False
        Whether to be used with the rate-full hierarchical likelihood.
    rate_key : string, default='rate'
        The key to recognize as N, where N is the total number of mergers in the Universe during the
        observing time, e.g., dN/d\theta = Np(\theta | \Lambda). This is only used if rate=True and 
        using bilby_model as bilby.hyper.model.Model, as this only returns probability densities.

    Returns
    -------
    callable
        A function with signature (data, parameters) -> probability values,
        where `data` is a dictionary of GW parameter samples and `parameters` are
        hyperparameters of the population model.
    """

    try:
        from gwpopulation.experimental.jax import NonCachingModel
    except ImportError:
        NonCachingModel = None
    valid_types = (bilby.hyper.model.Model,)
    if NonCachingModel is not None:
        valid_types = valid_types + (NonCachingModel,)
    if not isinstance(bilby_model, valid_types):
        # TODO: add some catches here, otherwise it assumes a particular form for the model
        return bilby_model # function of data, parameters

    from copy import copy
    copy_models = [copy(m) for m in bilby_model.models]
    bilby_model = bilby.hyper.model.Model(copy_models, cache=False)
    
    def model_to_return(data, parameters):
        if rate:
            R = parameters.pop(rate_key)
        else:
            R = 1.

        parameters, added_keys = conversion_function(parameters)
        bilby_model.parameters.update(parameters)
        return R*bilby_model.prob(data)
    
    return model_to_return

def _compute_mean_weights_for_correction(
        hyperposterior, n, bilby_model, gw_dataset, MC_integral_size=None, conversion_function=lambda args: (args, None), 
        MC_type='single event', verbose=True, rate=False, rate_key='rate'
        ):
    r"""
    Compute mean event or selection weights integrated over hyperposterior samples. 

    For a Monte Carlo integral
    
    \hat{I}(\Lambda) = \frac{1}{M}\sum_{i=1}^M \frac{p(\theta_i | \Lambda)}{p(\theta_i | {\rm draw})},

    and a set of $N_{\rm samp}$ samples from the hyperposterior $\Lambda_n$, then compute

    \overline{w}_i = \frac{1}{N_{\rm samp}} \sum_{n=1}^{N_{\rm samp}} \frac{1}{\hat{I}(\Lambda_n)}\frac{p(\theta_i | \Lambda_n)}{p(\theta_i | {\rm draw})},

    the weight averaged over the hyperposterior and dividing out $\hat{I}$, for use in computing the error statistics.

    
    Parameters
    ----------
    hyperposterior : dict of jnp.ndarray
        Hyperposterior samples with keys as hyperparameters, and values are jnp.ndarray 
        with first dimension indexing the hyperposterior sample
    n : int
        Number of samples in the hyperposterior
    bilby_model : bilby.hyper.model.Model, or callable
        Population model used to compute probabilities.
    gw_dataset : dict
        Dictionary of GW data samples, must include 'prior' key for sampling prior.
    MC_integral_size : int, optional
        Number of Monte Carlo samples. If None, inferred from `gw_dataset` or prior shape.
    conversion_function : callable, optional
        Function to convert hyperposterior parameters before model evaluation. For example, 
        gwpopulation.conversions.convert_to_beta_parameters
    MC_type : str, default='single event'
        Label for progress bar (e.g. 'single event' or 'selection').
    verbose : bool, default=True
        Whether to show a progress bar.
    rate: bool, default=False
        Whether to compute the integrated weight *without* dividing by the MC integral. For use with 
        the rate-full likelihood. If setting rate=True, then the bilby_model must return dN/d\theta. It 
        should *not* be in units of comoving merger rate density. 
    rate_key : string, default='rate'
        The key to recognize as N, where N is the total number of mergers in the Universe during the
        observing time, e.g., dN/d\theta = Np(\theta | \Lambda). This is only used if rate=True and 
        using bilby_model as bilby.hyper.model.Model, as this only returns probability densities.

    Returns
    -------
    jnp.ndarray
        Mean normalized event weights across hyperposterior samples,
        shape matching the sampling prior. 
    """

    model_function = bilby_model_to_model_function(bilby_model, conversion_function=conversion_function, rate=rate, rate_key=rate_key)

    gw_dataset = gw_dataset.copy()
    sampling_prior = gw_dataset.pop('prior')

    if MC_integral_size is None:
        try:
            MC_integral_size = gw_dataset.pop('total_generated')
        except KeyError:
            MC_integral_size = sampling_prior.shape[-1]

    mean_event_weights = jnp.zeros_like(sampling_prior) # (Nevents, NPE)
    
    keys = hyperposterior.keys()
    
    def weights_for_single_sample(ii, mean_event_weights):
        parameters = {k: hyperposterior[k][ii] for k in keys}
        weights = model_function(gw_dataset, parameters) / sampling_prior
        if rate:
            expectation = jnp.ones(weights.shape[:-1])
        else:
            expectation = jnp.sum(weights, axis=-1) / MC_integral_size
            
        return mean_event_weights + weights / expectation[..., None] / n

    if verbose:
        f = jax_tqdm.loop_tqdm(n, print_rate=1, tqdm_type='std', desc=f'Computing {MC_type} covariance weights integrated over hyperposterior samples')
    else:
        f = jax.jit

    weights_for_single_sample = f(weights_for_single_sample)

    mean_event_weights = jax.lax.fori_loop(
        0, 
        n, 
        weights_for_single_sample, 
        mean_event_weights,
        )

    return mean_event_weights

def _compute_integrated_cov(integrated_weights, sample, model_function, gw_dataset, MC_integral_size=None, rate=False):
    r"""
    Compute integrated covariance and variance for weights of a single posterior sample.

    Parameters
    ----------
    integrated_weights : jnp.ndarray
        Precomputed integrated weights across hyperposterior samples.
    sample : dict
        A single hyperposterior parameter sample.
    model_function : callable
        Function mapping (dataset, parameters) -> probability values.
    gw_dataset : dict
        Dataset dictionary with GW parameter samples, must include 'prior'.
    MC_integral_size : int, optional
        Number of Monte Carlo samples. If None, inferred from dataset.
    rate : bool, default=True
        Whether to assume rate-full likelihood. If True, then the weights are assumed to include the overall 
        rate normalization, N, where dN/d\theta = Np(\theta | \Lambda). It should only be set to True when
        computing the integrated covariance for the selection efficiency integral.

    Returns
    -------
    tuple of jnp.ndarray
        - integrated_cov : Integrated covariance estimate for the sample.
        - var : Variance estimate for the sample.
    """

    gw_dataset = gw_dataset.copy()
    sampling_prior = gw_dataset.pop('prior')

    if MC_integral_size is None:
        try:
            MC_integral_size = gw_dataset.pop('total_generated')
        except KeyError:
            MC_integral_size = sampling_prior.shape[-1]

    weights = model_function(gw_dataset, sample) / sampling_prior

    if rate:
        expectation = jnp.ones(weights.shape[:-1])
    else:
        expectation = jnp.sum(weights, axis=-1) / MC_integral_size

    var = (-1. + jnp.sum(weights**2, axis=-1) / MC_integral_size / expectation**2) / (MC_integral_size - 1)
    integrated_cov = (-1. + jnp.sum(integrated_weights * weights, axis=-1) / MC_integral_size / expectation) / (MC_integral_size - 1)

    return integrated_cov, var
    

def pad_ragged_posteriors(event_posteriors):
    """
    Pad a list of per-event posterior dicts into a single rectangular dict.

    Different events may have different numbers of posterior samples ("ragged").
    This stacks them into ``(Nobs, NPE_max)`` arrays so the rest of the vectorized
    machinery can run unchanged. Padded entries are made inert by setting their
    ``'prior'`` to ``+inf`` (so the importance weight ``model/prior`` is exactly 0);
    every other key is padded by repeating that event's first sample, which keeps the
    population model finite on the padded rows (avoiding ``inf/inf``).

    The returned ``counts`` array gives each event's real sample count and should be
    used as the Monte-Carlo integral size (instead of ``NPE_max``) so per-event means
    divide by the true number of samples.

    Parameters
    ----------
    event_posteriors : list of dict
        One dict per event, mapping parameter name to a 1-D ``(NPE_i,)`` array. Each
        dict must contain ``'prior'``.

    Returns
    -------
    padded : dict
        Each key mapped to a ``(Nobs, NPE_max)`` array.
    counts : jnp.ndarray
        Real per-event sample counts, shape ``(Nobs,)``.
    """
    counts = jnp.array([jnp.asarray(event['prior']).shape[0] for event in event_posteriors])
    npe_max = int(counts.max())
    keys = list(event_posteriors[0].keys())

    padded = {}
    for key in keys:
        rows = []
        for event in event_posteriors:
            arr = jnp.asarray(event[key])
            pad_len = npe_max - arr.shape[0]
            if pad_len > 0:
                if key == 'prior':
                    fill = jnp.full(pad_len, jnp.inf, dtype=arr.dtype)
                else:
                    # repeat a real, in-support sample so the model stays finite
                    fill = jnp.full(pad_len, arr[0], dtype=arr.dtype)
                arr = jnp.concatenate([arr, fill])
            rows.append(arr)
        padded[key] = jnp.stack(rows)

    return padded, counts

def format_hyperposterior(hyperposterior):
    if isinstance(hyperposterior, pd.DataFrame):
        hyperposterior = hyperposterior.to_dict(orient='list')
    else:
        if not isinstance(hyperposterior, dict):
            raise IOError(f"Hyperposterior must be a dictionary or pandas.DataFrame, not {type(hyperposterior)}")

    ns = []
    for k in hyperposterior.keys():
        hyperposterior[k] = jnp.array(hyperposterior[k])
        ns.append(hyperposterior[k].shape[0])

    if not jnp.all(jnp.array(ns) == ns[0]):
        raise IOError(f"Hyperposterior has unequal number of samples for hyperparameters.")

    n = ns[0]
    return hyperposterior, n

def _prepare_error_statistics_inputs(event_posteriors, hyperposterior, nobs, event_counts, verbose):
    """
    Pad ragged event posteriors, format the hyperposterior, and infer Nobs if needed.
    """

    # Ragged input: a list of per-event dicts. Pad to a rectangular dict and use the
    # real per-event counts as the single-event Monte-Carlo integral size.
    if isinstance(event_posteriors, (list, tuple)):
        event_posteriors, event_counts = pad_ragged_posteriors(event_posteriors)

    hyperposterior, n = format_hyperposterior(hyperposterior)

    if nobs is None:
        nobs = event_posteriors['prior'].shape[0]
        if verbose:
            print(f'Nobs not provided, assuming Nobs = {nobs}')
    return event_posteriors, event_counts, hyperposterior, n, nobs

def _integrated_covariances(
        model_function, vt_model_function, injections, event_posteriors, hyperposterior, n,
        conversion_function, verbose, rate, rate_key, event_counts,
        ):
    """
    For each hyperposterior sample Lambda_n, compute the single-event and selection log-variances
    and their covariances averaged over all other hyperposterior samples, without storing the
    weights for every sample.

    Returns
    -------
    tuple of jnp.ndarray
        (event_integrated_covs, event_vars) of shape (n, Nobs) and (vt_integrated_covs, vt_vars) of shape (n,).
    """

    total_generated = injections['total_generated']

    mean_event_weights = _compute_mean_weights_for_correction(
        hyperposterior,
        n,
        model_function,
        event_posteriors,
        MC_integral_size=event_counts,
        conversion_function=conversion_function,
        MC_type='single event',
        verbose=verbose
        )
    mean_vt_weights = _compute_mean_weights_for_correction(
        hyperposterior, 
        n,
        vt_model_function, 
        injections, 
        MC_integral_size=total_generated, 
        conversion_function=conversion_function, 
        MC_type='selection', 
        verbose=verbose,
        rate=rate,
        rate_key=rate_key
        )

    def create_loop_fn(m, p, MC_type='single event', MC_integral_size=None):
        if MC_type=='single event':
            _rate = False
            _model_function = bilby_model_to_model_function(model_function, conversion_function=conversion_function, rate=_rate, rate_key=rate_key)
        else:
            _rate = rate
            _model_function = bilby_model_to_model_function(vt_model_function, conversion_function=conversion_function, rate=_rate, rate_key=rate_key)
        loop_fn = lambda _, sample: (_, (sample[0],)+ _compute_integrated_cov(
                m,
                sample[1],
                _model_function,
                p,
                MC_integral_size=MC_integral_size,
                rate=_rate
                ))
        if verbose:
            return jax_tqdm.scan_tqdm(n, print_rate=1, tqdm_type='std', desc=f'For each posterior sample, average {MC_type} covariance with another posterior sample')(loop_fn)
        else:
            return jax.jit(loop_fn)

    _, (_, event_integrated_covs, event_vars) = jax.lax.scan(
        create_loop_fn(mean_event_weights, event_posteriors, MC_integral_size=event_counts),
        0,
        (jnp.arange(n), hyperposterior),
        length=n
    )
    
    _, (_, vt_integrated_covs, vt_vars) = jax.lax.scan(
        create_loop_fn(mean_vt_weights, injections, MC_type='selection'),
        0,
        (jnp.arange(n), hyperposterior),
        length=n
    )
    return event_integrated_covs, event_vars, vt_integrated_covs, vt_vars

def error_statistics(
        model_function,
        injections,
        event_posteriors,
        hyperposterior,
        vt_model_function=None,
        include_likelihood_correction=True,
        conversion_function=lambda args: (args, None),
        nobs=None,
        verbose=True,
        rate=False,
        rate_key='rate',
        event_counts=None,
        ):
    """
    Compute error, precision, and accuracy statistics from model, hyperposterior, and data.

    Parameters
    ----------
    model_function : bilby.hyper.model.Model, callable
        Population model with interface (dataset, parameters) -> probabilities.
    injections : dict
        Injection dataset, including 'prior' and 'total_generated' keys.
    event_posteriors : dict or list of dict
        Event posterior samples, including 'prior' key. Either a rectangular dict with
        ``(Nobs, NPE)`` arrays, or a *list of per-event dicts* with 1-D ``(NPE_i,)``
        arrays (ragged: events may have different sample counts). A list is padded
        internally via :func:`pad_ragged_posteriors` and the real per-event counts are
        used as the Monte-Carlo integral size.
    hyperposterior : pandas.DataFrame or dict of jnp.ndarray
        If pandas.DataFrame, converts to appropriate format. Otherwise, hyperposterior 
        samples with keys as hyperparameters, and values are jnp.ndarray with first 
        dimension indexing the hyperposterior sample
    vt_model_function : bilby.hyper.model.Model, callable, optional
        Optional separate model instance for evaluating the selection function. 
        Population model with interface (dataset, parameters) -> probabilities. If not 
        included, set to model_function
    include_likelihood_correction : bool, default=True
        Whether to include likelihood correction in accuracy estimate. 
        Set to False if the hyperlikelihood for sampling from the posterior was estimated
        using the unbiased likelihood of Eq. 24 of https://arxiv.org/abs/2509.07221
    conversion_function : callable, optional
        Function to convert hyperposterior parameters before model evaluation.
    nobs : int, optional
        Number of observed events. If None, inferred from `event_posteriors`.
    verbose : bool, default=True
        Whether to print progress and summary messages.
    rate : bool, default=False
        Whether to treat the VT weights as rate-weighted. TESTTHIS!!!
    rate_key : string, default='rate'
        The key which to access the overall merger rate within the posterior.
    event_counts : jnp.ndarray, optional
        Per-event real sample counts, shape ``(Nobs,)``, used as the single-event
        Monte-Carlo integral size. Set automatically when ``event_posteriors`` is a
        ragged list; pass explicitly if you pre-pad a rectangular dict yourself. If
        ``None`` and ``event_posteriors`` is rectangular, ``NPE`` is used.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'error_statistic' : float, total information loss in bits.
        - 'precision_statistic' : float, information loss due to variance.
        - 'accuracy_statistic' : float, information loss due to bias.
    """

    event_posteriors, event_counts, hyperposterior, n, nobs = _prepare_error_statistics_inputs(
        event_posteriors, hyperposterior, nobs, event_counts, verbose
        )
    if vt_model_function is None:
        vt_model_function = model_function

    event_integrated_covs, event_vars, vt_integrated_covs, vt_vars = _integrated_covariances(
        model_function, vt_model_function, injections, event_posteriors, hyperposterior, n,
        conversion_function, verbose, rate, rate_key, event_counts,
        )

    if rate:
        nobs = 1
    var = jnp.sum(event_vars, axis=-1) + nobs**2 * vt_vars
    cov = jnp.sum(event_integrated_covs, axis=-1) + nobs**2 * vt_integrated_covs

    event_precision = float(jnp.mean(jnp.sum(event_vars - event_integrated_covs, axis=-1)) / 2 / jnp.log(2))
    vt_precision = float(nobs**2 * jnp.mean(vt_vars - vt_integrated_covs) / 2 / jnp.log(2))
    
    precision = float((jnp.mean(var) - jnp.mean(cov)) / 2 / jnp.log(2))
    if include_likelihood_correction:
        if rate:
            correction = vt_vars / 2
        else:
            correction = nobs*(nobs+1) * vt_vars / 2
        accuracy = float(jnp.var(cov - correction) / 2 / jnp.log(2))
        selection_w = nobs**2 * vt_integrated_covs - correction
    else:
        accuracy = float(jnp.var(cov) / 2 / jnp.log(2))
        selection_w = nobs**2 * vt_integrated_covs
    event_w = jnp.sum(event_integrated_covs, axis=-1)
    event_accuracy = float(jnp.var(event_w) / 2 / jnp.log(2))
    selection_accuracy = float(jnp.var(selection_w) / 2 / jnp.log(2))
    correlation_accuracy = float(jnp.mean((event_w - jnp.mean(event_w))*(selection_w - jnp.mean(selection_w))) / jnp.log(2))
    
    error = float(precision + accuracy)
    
    if verbose:
        print(f'\nYour inference loses approximately {round(error, 3)} bits of information to Monte Carlo approximations.')
        print(f'Of the total information loss')
        print(f' * {round(precision, 3)} bits is from uncertainty in the posterior. Of this')
        print(f'    * {round(100*event_precision/precision, 1)}% is from the single-event Monte Carlo integration')
        print(f'    * {round(100*vt_precision/precision, 1)}% is from the selection Monte Carlo integration')
        print(f' * {round(accuracy, 5)} bits is from bias in the posterior. Of the total bias')
        print(f'    * {round(100*event_accuracy/accuracy, 1)}% is from the single-event Monte Carlo integration')
        print(f'    * {round(100*selection_accuracy/accuracy, 1)}% is from the selection Monte Carlo integration')
        print(f'    * {round(100*correlation_accuracy/accuracy, 1)}% is from correlations in the uncertainty of the single-event and selection MC integrals')
    
    # how much due to VT and how much due to events? We can also compute this :O I believe bc they are additive. Well, 
    # I don't know if we can do it necessarily for the accuracy statistic, because Var(E + V) = Var(E) + 2Cov(E,V) + Var(V), so 
    # we would technically have a "covariance" between event and vt terms. Still, could be interesting at least to compute precision from VT and precision from events
    
    return {
        'error_statistic': error, 
        'precision_statistic': precision, 
        'accuracy_statistic': accuracy,
        'event_precision_statistic': event_precision,
        'selection_precision_statistic': vt_precision,
        'event_accuracy_statistic': event_accuracy,
        'selection_accuracy_statistic': selection_accuracy,
        'correlation_event_selection_accuracy_statistic': correlation_accuracy,
        }

def marginal_error_statistics(
        model_function,
        injections,
        event_posteriors,
        hyperposterior,
        parameters=None,
        k_neighbours=1,
        vt_model_function=None,
        include_likelihood_correction=True,
        conversion_function=lambda args: (args, None),
        nobs=None,
        verbose=True,
        rate=False,
        rate_key='rate',
        event_counts=None,
        ):
    """
    Compute error statistics for the one-dimensional marginal hyperposteriors, without storing
    weights for every hyperposterior sample.

    This is the memory-efficient analogue of :func:`marginal_error_statistics_from_weights`; see
    there for the method. For each parameter, the distinct hyperposterior samples are visited in
    order of that parameter while keeping the normalized weights of the previous K samples, so
    every nearest-neighbour covariance is available with one model evaluation per sample. The
    cost is therefore ~(2 + n_parameters) model evaluations per hyperposterior sample, and the
    memory is ~K times that of a single sample's weights (plus one per extra copy of a repeated sample).

    Parameters
    ----------
    model_function, injections, event_posteriors, hyperposterior, vt_model_function,
    include_likelihood_correction, conversion_function, nobs, verbose, rate, rate_key, event_counts
        As in :func:`error_statistics`. Repeated hyperposterior samples, including copies that
        differ only in post-processed columns such as a sampled rate, are identified automatically.
    parameters : list of str, optional
        Hyperparameters for which to compute marginal statistics. Defaults to all columns.
    k_neighbours : int, default=1
        Number of nearest neighbours in x averaged over for each sample. Larger K reduces the
        noise of the estimate at the cost of resolution in x.

    Returns
    -------
    dict
        - 'joint' : dict of the joint (error, precision, accuracy) statistics.
        - 'marginal' : dict mapping each parameter to its dict of (error, precision, accuracy)
          statistics. Parameters that take a single value or have non-finite values are mapped
          to NaNs. The estimates are noisy and can be slightly negative when the true value is ~0.
    """

    event_posteriors, event_counts, hyperposterior, n, nobs = _prepare_error_statistics_inputs(
        event_posteriors, hyperposterior, nobs, event_counts, verbose
        )
    if vt_model_function is None:
        vt_model_function = model_function
    keys, representative, multiplicity = _distinct_samples(hyperposterior)
    tiebreak = np.random.default_rng(0).random(len(representative))
    if parameters is None:
        parameters = keys

    event_integrated_covs, event_vars, vt_integrated_covs, vt_vars = _integrated_covariances(
        model_function, vt_model_function, injections, event_posteriors, hyperposterior, n,
        conversion_function, verbose, rate, rate_key, event_counts,
        )

    if rate:
        nobs = 1
    var = np.asarray(jnp.sum(event_vars, axis=-1) + nobs**2 * vt_vars)
    cov = np.asarray(jnp.sum(event_integrated_covs, axis=-1) + nobs**2 * vt_integrated_covs)
    if not include_likelihood_correction:
        correction = np.zeros(n)
    elif rate:
        correction = np.asarray(vt_vars) / 2
    else:
        correction = nobs * (nobs + 1) * np.asarray(vt_vars) / 2

    joint_mean_cov = float(np.mean(cov))
    joint_precision = float((np.mean(var) - joint_mean_cov) / 2 / np.log(2))
    joint_accuracy = float(np.var(cov - correction) / 2 / np.log(2))
    joint = {'error_statistic': joint_precision + joint_accuracy, 'precision_statistic': joint_precision, 'accuracy_statistic': joint_accuracy}

    b = (cov - correction)[representative]
    groups = _copy_groups(np.asarray(event_vars)[representative])
    half_width = _window_half_width(groups, k_neighbours)

    event_data = dict(event_posteriors)
    event_prior = event_data.pop('prior')
    event_size = event_prior.shape[-1] if event_counts is None else event_counts
    vt_data = dict(injections)
    vt_prior = vt_data.pop('prior')
    vt_size = vt_data.pop('total_generated', vt_prior.shape[-1])
    def normalized_weights(model, data, prior, size, sample, _rate):
        weights = model(data, dict(sample)) / prior
        if _rate:
            return weights
        return weights / (jnp.sum(weights, axis=-1) / size)[..., None]

    def create_loop_fn():
        # bilby models hold their parameters as state, so build fresh ones for every scan to avoid leaking tracers
        event_model = bilby_model_to_model_function(model_function, conversion_function=conversion_function, rate=False, rate_key=rate_key)
        vt_model = bilby_model_to_model_function(vt_model_function, conversion_function=conversion_function, rate=rate, rate_key=rate_key)

        def neighbour_covariances(buffers, xs):
            # covariance of ln L between this sample and each of the previous half_width samples in sorted order
            event_buffer, vt_buffer = buffers
            _, sample = xs
            event_w = normalized_weights(event_model, event_data, event_prior, event_size, sample, False)
            vt_w = normalized_weights(vt_model, vt_data, vt_prior, vt_size, sample, rate)
            event_cov = (jnp.sum(event_buffer * event_w, axis=-1) / event_size - 1) / (event_size - 1)
            vt_cov = (jnp.sum(vt_buffer * vt_w, axis=-1) / vt_size - 1) / (vt_size - 1)
            buffers = (
                jnp.concatenate([event_w[None], event_buffer[:-1]]),
                jnp.concatenate([vt_w[None], vt_buffer[:-1]]),
                )
            return buffers, jnp.sum(event_cov, axis=-1) + nobs**2 * vt_cov
        return neighbour_covariances

    marginal = {}
    for key in parameters:
        x = np.asarray(hyperposterior[key])[representative]
        if _invalid_marginal_parameter(x, key):
            marginal[key] = {'error_statistic': np.nan, 'precision_statistic': np.nan, 'accuracy_statistic': np.nan}
            continue
        order, offsets, neighbour_weights = _marginal_nearest_neighbours(x, multiplicity, groups, k_neighbours, tiebreak)
        U = len(order)
        sorted_samples = {k: hyperposterior[k][representative[order]] for k in keys}

        if verbose:
            loop_fn = jax_tqdm.scan_tqdm(U, print_rate=1, tqdm_type='std', desc=f'Nearest-neighbour covariances in {key}')(create_loop_fn())
        else:
            loop_fn = jax.jit(create_loop_fn())
        buffers = (jnp.zeros((half_width,) + event_prior.shape), jnp.zeros((half_width,) + vt_prior.shape))
        _, previous_covariances = jax.lax.scan(loop_fn, buffers, (jnp.arange(U), sorted_samples), length=U)
        previous_covariances = np.asarray(previous_covariances) # [p, j] = Cov between positions p and p - 1 - j

        positions = np.arange(U)[:, None]
        later = offsets > 0
        rows = np.clip(np.where(later, positions + offsets, positions), 0, U - 1)
        cols = np.clip(np.abs(offsets) - 1, 0, half_width - 1)
        pair_covariance = np.where(neighbour_weights > 0, previous_covariances[rows, cols], 0.)

        marginal[key] = _marginal_statistics(pair_covariance, b, multiplicity, order, offsets, neighbour_weights, n, joint_mean_cov)

    if verbose:
        _print_marginal_statistics(joint, marginal)
    return {'joint': joint, 'marginal': marginal}

def _log_likelihood_covariance_matrix(
        model_function, vt_model_function, injections, event_posteriors, hyperposterior, representative, multiplicity, n,
        conversion_function, nobs, rate, rate_key, event_counts, block_size, sketch_size, seed, verbose,
        ):
    """
    Covariance of ln L between all pairs of distinct hyperposterior samples, centred on the
    hyperposterior mean.

    Cov[ln L(Lambda_n), ln L(Lambda_m)] = <v_n, v_m> - c_0 is an inner product of per-sample
    feature vectors v_n (the normalized single-event and selection weights, scaled by
    1/sqrt(M (M-1))). With vbar the hyperposterior mean of v_n, this returns

        G[n, m] = <v_n - vbar, v_m - vbar>,    a[n] = <v_n - vbar, vbar>,

    from which Cov[n, m] = G[n, m] + a[n] + a[m] + const. Every error statistic only needs
    differences of covariances, so the constant (and the cancellation against c_0) never appears.

    The features have length Nobs * NPE + Ninj and are never stored for every sample. Exactly,
    G is built in blocks of block_size samples, costing ~U + U^2 / (2 block_size) model
    evaluations. With sketch_size, each feature vector is compressed by a CountSketch into
    sketch_size dimensions in a single pass (~2U evaluations), giving an unbiased estimate of the
    off-diagonal of G with relative error ~1/sqrt(sketch_size); the diagonal is always exact.

    Returns
    -------
    G : np.ndarray, shape (U, U)
    a : np.ndarray, shape (U,)
    vt_vars : np.ndarray, shape (U,), Var[ln of the selection MC integral], for the likelihood correction
    event_vars : np.ndarray, shape (U, Nobs), Var[ln of each single-event MC integral], to identify copies
    """
    from tqdm import tqdm

    event_data = dict(event_posteriors)
    event_prior = jnp.asarray(event_data.pop('prior'))
    Nobs = event_prior.shape[0]
    event_size = jnp.broadcast_to(jnp.asarray(event_prior.shape[-1] if event_counts is None else event_counts, dtype=event_prior.dtype), (Nobs,))
    vt_data = dict(injections)
    vt_prior = jnp.asarray(vt_data.pop('prior'))
    vt_size = vt_data.pop('total_generated', vt_prior.shape[-1])
    event_scale = 1 / jnp.sqrt(event_size * (event_size - 1))
    vt_scale = nobs / jnp.sqrt(vt_size * (vt_size - 1.))

    U = len(representative)
    if block_size is None:
        # aim for ~0.5GB of features per block
        block_size = max(1, int(5e8 // ((event_prior.size + vt_prior.size) * event_prior.dtype.itemsize)))
    block_size = min(block_size, U)
    blocks = [np.arange(start, min(start + block_size, U)) for start in range(0, U, block_size)]

    def create_sample_weights():
        # bilby models hold their parameters as state, so build fresh ones for every traced function
        event_model = bilby_model_to_model_function(model_function, conversion_function=conversion_function, rate=False, rate_key=rate_key)
        vt_model = bilby_model_to_model_function(vt_model_function, conversion_function=conversion_function, rate=rate, rate_key=rate_key)

        def sample_weights(sample):
            event_w = event_model(event_data, dict(sample)) / event_prior
            event_w = event_w / (jnp.sum(event_w, axis=-1) / event_size)[..., None]
            vt_w = vt_model(vt_data, dict(sample)) / vt_prior
            if not rate:
                vt_w = vt_w / (jnp.sum(vt_w) / vt_size)
            vt_var = (jnp.sum(vt_w**2) / vt_size - 1) / (vt_size - 1)
            event_var = (jnp.sum(event_w**2, axis=-1) / event_size - 1) / (event_size - 1)
            return event_w, vt_w, vt_var, event_var
        return sample_weights

    def block_inputs(block):
        # pad the last block to block_size so each function is only traced once; padding has zero multiplicity
        index = np.concatenate([block, np.full(block_size - len(block), block[-1])])
        copies = np.concatenate([multiplicity[block], np.zeros(block_size - len(block))])
        return {k: v[representative[index]] for k, v in hyperposterior.items()}, jnp.asarray(copies, dtype=event_prior.dtype)

    # first pass: hyperposterior mean of the normalized weights
    sample_weights = create_sample_weights()
    def accumulate(sums, xs):
        sample, copies = xs
        event_w, vt_w, vt_var, event_var = sample_weights(sample)
        return (sums[0] + copies * event_w, sums[1] + copies * vt_w), (vt_var, event_var)
    mean_fn = jax.jit(lambda samples, copies: jax.lax.scan(accumulate, (jnp.zeros_like(event_prior), jnp.zeros_like(vt_prior)), (samples, copies)))

    event_mean, vt_mean, vt_vars, event_vars = 0., 0., np.zeros(U), np.zeros((U, Nobs))
    for block in tqdm(blocks, desc='Hyperposterior mean of the weights', disable=not verbose):
        (event_sum, vt_sum), (vt_var, event_var) = mean_fn(*block_inputs(block))
        event_mean, vt_mean = event_mean + event_sum / n, vt_mean + vt_sum / n
        vt_vars[block] = np.asarray(vt_var)[:len(block)]
        event_vars[block] = np.asarray(event_var)[:len(block)]
    mean_features = jnp.concatenate([(event_mean * event_scale[:, None]).ravel(), vt_scale * vt_mean])

    def create_sample_features():
        sample_weights = create_sample_weights()
        def sample_features(sample):
            event_w, vt_w, _, _ = sample_weights(sample)
            return jnp.concatenate([((event_w - event_mean) * event_scale[:, None]).ravel(), vt_scale * (vt_w - vt_mean)])
        return sample_features

    G = np.zeros((U, U))
    a = np.zeros(U)
    if sketch_size is None:
        sample_features = create_sample_features()
        features_fn = jax.jit(lambda samples: jax.lax.map(sample_features, samples))
        gram = jax.jit(lambda f1, f2: jax.lax.dot_general(f1, f2, (((1,), (1,)), ((), ()))))
        n_evaluations = len(blocks) * (len(blocks) + 1) // 2
        with tqdm(total=n_evaluations, desc='Covariance matrix blocks', disable=not verbose) as progress:
            for i, block_i in enumerate(blocks):
                features_i = features_fn(block_inputs(block_i)[0])
                a[block_i] = np.asarray(features_i @ mean_features)[:len(block_i)]
                G[np.ix_(block_i, block_i)] = np.asarray(gram(features_i, features_i))[:len(block_i), :len(block_i)]
                progress.update()
                for block_j in blocks[i + 1:]:
                    G_ij = np.asarray(gram(features_i, features_fn(block_inputs(block_j)[0])))[:len(block_i), :len(block_j)]
                    G[np.ix_(block_i, block_j)] = G_ij
                    G[np.ix_(block_j, block_i)] = G_ij.T
                    progress.update()
                del features_i
    else:
        rng = np.random.default_rng(seed)
        L = mean_features.shape[0]
        buckets = jnp.asarray(rng.integers(0, sketch_size, L))
        signs = jnp.asarray(rng.choice([-1., 1.], L), dtype=mean_features.dtype)
        sample_features = create_sample_features()
        def sample_sketch(sample):
            features = sample_features(sample)
            return jax.ops.segment_sum(features * signs, buckets, num_segments=sketch_size), features @ mean_features, features @ features
        sketch_fn = jax.jit(lambda samples: jax.lax.map(sample_sketch, samples))

        sketches, norms = jnp.zeros((U, sketch_size), dtype=mean_features.dtype), np.zeros(U)
        for block in tqdm(blocks, desc='Sketching covariance features', disable=not verbose):
            block_sketch, block_a, block_norms = sketch_fn(block_inputs(block)[0])
            sketches = sketches.at[block].set(block_sketch[:len(block)])
            a[block] = np.asarray(block_a)[:len(block)]
            norms[block] = np.asarray(block_norms)[:len(block)]
        G = np.array(jax.lax.dot_general(sketches, sketches, (((1,), (1,)), ((), ()))))
        G[np.diag_indices(U)] = norms
    return G, a, vt_vars, event_vars

@partial(jax.jit, static_argnums=(8, 9))
def _nearest_neighbour_marginals(x, G, a, bias, multiplicity, groups, n, tiebreak, k_neighbours, half_width):
    """
    Marginal (precision, accuracy) statistics for a batch of hyperparameters x, shape (batch, U),
    from the centred covariance matrix G, the offsets a and the centred bias of each distinct
    sample (see :func:`_log_likelihood_covariance_matrix`); copy groups and tiebreak are as in
    :func:`_marginal_nearest_neighbours`. The arrays are arguments rather than
    closed over so that they are not compiled into the function as constants.
    """
    U = G.shape[0]

    def single(x):
        order = jnp.lexsort((tiebreak, x))
        offsets, weights = _sorted_window_neighbours(x[order], multiplicity[order], groups[order], k_neighbours, half_width)
        partners = order[jnp.clip(jnp.arange(U)[:, None] + offsets, 0, U - 1)]
        outer = multiplicity[order][:, None] * weights / n / k_neighbours
        precision = jnp.sum(outer * (G[order[:, None], partners] + a[partners]))
        accuracy = jnp.sum(outer * bias[order][:, None] * bias[partners])
        return precision / 2 / jnp.log(2), accuracy / 2 / jnp.log(2)

    return jax.vmap(single)(x)

def marginal_error_statistics_matrix(
        model_function,
        injections,
        event_posteriors,
        hyperposterior,
        parameters=None,
        k_neighbours=1,
        vt_model_function=None,
        include_likelihood_correction=True,
        conversion_function=lambda args: (args, None),
        nobs=None,
        verbose=True,
        rate=False,
        rate_key='rate',
        event_counts=None,
        block_size=None,
        sketch_size=None,
        dimension_batch_size=128,
        null_replicates=100,
        seed=0,
        covariance=None,
        return_covariance=False,
        ):
    """
    Compute error statistics for the one-dimensional marginal hyperposteriors of many
    hyperparameters at once.

    The estimator is the same as :func:`marginal_error_statistics` (see
    :func:`marginal_error_statistics_from_weights` for the method), but the model evaluations are
    shared between all hyperparameters: the covariance of ln L between every pair of distinct
    hyperposterior samples is computed once as a (U, U) matrix, after which each marginal only
    needs a sort and a gather of n * K matrix entries. The marginals are computed in batches of
    hyperparameters on the device, so this scales to very many hyperparameters. The matrix needs
    U^2 floats of memory (~1GB in float64 for U ~ 10^4 distinct samples).

    Also returns a null distribution for the marginal statistics, from applying the same estimator
    to random, independent hyperparameters. This is the distribution of the estimate for a
    hyperparameter that the Monte Carlo noise does not depend on (true marginal statistics of 0),
    so its spread is the noise floor: with many hyperparameters, only marginals well above it are
    meaningful.

    Parameters
    ----------
    model_function, injections, event_posteriors, hyperposterior, vt_model_function,
    include_likelihood_correction, conversion_function, nobs, verbose, rate, rate_key, event_counts
        As in :func:`error_statistics`. Repeated hyperposterior samples, including copies that
        differ only in post-processed columns such as a sampled rate, are identified automatically.
    parameters : list of str, optional
        Hyperparameters for which to compute marginal statistics. Defaults to all columns.
    k_neighbours : int, default=1
        Number of nearest neighbours in x averaged over for each sample.
    block_size : int, optional
        Number of hyperposterior samples whose weights are held in memory at once when building the
        exact covariance matrix. The matrix costs ~U + U^2 / (2 block_size) model evaluations.
        Defaults to ~0.5GB of weights. With sketch_size, only sets how many samples are processed per call.
    sketch_size : int, optional
        If given, approximate the covariance matrix with a CountSketch of this many dimensions,
        which costs ~2U model evaluations. The off-diagonal entries then have relative error
        ~1/sqrt(sketch_size). Use when U is too large for the exact matrix.
    dimension_batch_size : int, default=128
        Number of hyperparameters whose marginals are computed together on the device. Memory
        is ~dimension_batch_size * U * 2K floats.
    null_replicates : int, default=100
        Number of random hyperparameters drawn for the null distribution. 0 disables it.
    seed : int, default=0
        Seed for the sketch and the null distribution.
    covariance : dict, optional
        The 'covariance' entry of a previous call with return_covariance=True, for the same
        hyperposterior and inputs. Skips all model evaluations, e.g. to compute marginals for
        another set of parameters or another k_neighbours.
    return_covariance : bool, default=False
        Whether to include the covariance matrix and related arrays in the output.

    Returns
    -------
    dict
        - 'joint' : dict of the joint (error, precision, accuracy) statistics.
        - 'marginal' : pandas.DataFrame indexed by parameter, with columns 'error_statistic',
          'precision_statistic' and 'accuracy_statistic', and, if the null distribution is computed,
          'precision_significance': the number of null standard deviations by which the precision
          statistic exceeds the null mean. Parameters that take a single value or have non-finite
          values are NaN. Estimates
          are noisy and can be slightly negative when the true value is ~0.
        - 'null' : dict with the mean and standard deviation of the precision and accuracy
          statistics of random hyperparameters ('precision_mean', 'precision_std', 'accuracy_mean',
          'accuracy_std'), and the replicates themselves ('precision', 'accuracy').
        - 'covariance' : only if return_covariance=True.
    """

    # keep the samples on the host as numpy columns; with very many hyperparameters, converting
    # through Python lists or holding every column on the device is the bottleneck
    if isinstance(hyperposterior, pd.DataFrame):
        hyperposterior_np = {k: hyperposterior[k].to_numpy() for k in hyperposterior.columns}
    else:
        hyperposterior_np = {k: np.asarray(v) for k, v in hyperposterior.items()}
    keys = list(hyperposterior_np.keys())
    if parameters is None:
        parameters = keys

    if covariance is None:
        event_posteriors, event_counts, hyperposterior, n, nobs = _prepare_error_statistics_inputs(
            event_posteriors, dict(hyperposterior_np), nobs, event_counts, verbose
            )
        if vt_model_function is None:
            vt_model_function = model_function
        _, representative, multiplicity = _distinct_samples(hyperposterior_np, keys)
        if rate:
            nobs = 1
        G, a, vt_vars, event_vars = _log_likelihood_covariance_matrix(
            model_function, vt_model_function, injections, event_posteriors, hyperposterior, representative, multiplicity, n,
            conversion_function, nobs, rate, rate_key, event_counts, block_size, sketch_size, seed, verbose,
            )
        if not include_likelihood_correction:
            correction = np.zeros_like(vt_vars)
        elif rate:
            correction = vt_vars / 2
        else:
            correction = nobs * (nobs + 1) * vt_vars / 2
        covariance = {
            'G': G, 'a': a, 'correction': correction, 'groups': _copy_groups(event_vars),
            'representative': representative, 'multiplicity': multiplicity, 'n': n,
            }
    G, a, correction, groups = covariance['G'], covariance['a'], covariance['correction'], covariance['groups']
    representative, multiplicity, n = covariance['representative'], covariance['multiplicity'], covariance['n']
    if len(hyperposterior_np[keys[0]]) != n:
        raise ValueError(f"covariance was computed from {n} hyperposterior samples, but the hyperposterior has {len(hyperposterior_np[keys[0]])}.")
    U = len(representative)
    if n - np.bincount(groups, weights=multiplicity).max() < k_neighbours:
        raise ValueError(f'Fewer than k_neighbours={k_neighbours} distinct hyperposterior samples available.')

    # per-sample bias, centred on its hyperposterior mean; a has hyperposterior mean 0
    bias = a - correction
    bias = bias - np.sum(multiplicity * bias) / n
    joint_precision = float(np.sum(multiplicity * np.diag(G)) / n / 2 / np.log(2))
    joint_accuracy = float(np.sum(multiplicity * bias**2) / n / 2 / np.log(2))
    joint = {'error_statistic': joint_precision + joint_accuracy, 'precision_statistic': joint_precision, 'accuracy_statistic': joint_accuracy}

    tiebreak = jnp.asarray(np.random.default_rng(seed).random(U))
    arrays = (jnp.asarray(G), jnp.asarray(a), jnp.asarray(bias), jnp.asarray(multiplicity), jnp.asarray(groups), n, tiebreak)
    half_width = _window_half_width(groups, k_neighbours)
    batch_statistics = lambda x: _nearest_neighbour_marginals(x, *arrays, k_neighbours, half_width)

    precision = np.full(len(parameters), np.nan)
    accuracy = np.full(len(parameters), np.nan)
    from tqdm import tqdm
    for start in tqdm(range(0, len(parameters), dimension_batch_size), desc='Marginal statistics', disable=not verbose):
        batch = parameters[start:start + dimension_batch_size]
        x = np.stack([hyperposterior_np[k][representative] for k in batch])
        # pad to a fixed batch size so the function is only traced once
        x = np.concatenate([x, np.repeat(x[-1:], dimension_batch_size - len(batch), axis=0)])
        batch_precision, batch_accuracy = batch_statistics(jnp.asarray(x))
        precision[start:start + len(batch)] = np.asarray(batch_precision)[:len(batch)]
        accuracy[start:start + len(batch)] = np.asarray(batch_accuracy)[:len(batch)]
    invalid = np.array([_invalid_marginal_parameter(hyperposterior_np[k][representative], k) for k in parameters], dtype=bool)
    precision[invalid], accuracy[invalid] = np.nan, np.nan
    marginal = pd.DataFrame(
        {'error_statistic': precision + accuracy, 'precision_statistic': precision, 'accuracy_statistic': accuracy},
        index=pd.Index(parameters, name='parameter'),
        )

    null = {}
    if null_replicates > 0:
        # the same estimator applied to random hyperparameters, which the MC noise cannot depend on
        null_precision, null_accuracy = [], []
        for key in jax.random.split(jax.random.PRNGKey(seed), -(-null_replicates // dimension_batch_size)):
            # one value per copy group, as for a sampled hyperparameter
            batch_precision, batch_accuracy = batch_statistics(jax.random.uniform(key, (dimension_batch_size, groups.max() + 1))[:, groups])
            null_precision.append(np.asarray(batch_precision))
            null_accuracy.append(np.asarray(batch_accuracy))
        null_precision = np.concatenate(null_precision)[:null_replicates]
        null_accuracy = np.concatenate(null_accuracy)[:null_replicates]
        null = {
            'precision_mean': float(np.mean(null_precision)), 'precision_std': float(np.std(null_precision)),
            'accuracy_mean': float(np.mean(null_accuracy)), 'accuracy_std': float(np.std(null_accuracy)),
            'precision': null_precision, 'accuracy': null_accuracy,
            }
        marginal['precision_significance'] = (marginal['precision_statistic'] - null['precision_mean']) / null['precision_std']

    if verbose:
        shown = marginal.sort_values('precision_statistic', ascending=False)
        if len(shown) > 20:
            print(f'\nShowing the 20 of {len(shown)} marginals with the largest precision statistic.')
            shown = shown.iloc[:20]
        _print_marginal_statistics(joint, shown[['error_statistic', 'precision_statistic', 'accuracy_statistic']].to_dict(orient='index'))
        if null:
            print(f"Null (random hyperparameter) precision statistic: {null['precision_mean']:.3g} ± {null['precision_std']:.3g} bits")

    result = {'joint': joint, 'marginal': marginal, 'null': null}
    if return_covariance:
        result['covariance'] = covariance
    return result
