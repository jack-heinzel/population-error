import jax.numpy as jnp
import jax
import jax_tqdm
import bilby
import gwpopulation

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

def reweighted_event_bayes_factors(event_pe_weights):
    """
    Compute reweighted Bayes factors for a set of events.

    Parameters
    ----------
    event_pe_weights : jnp.ndarray
        Array of shape (Nobs, NPE) with posterior sample weights per event.

    Returns
    -------
    jnp.ndarray
        Array of mean Bayes factors per event, shape (Nobs,).
    """
    
    return jnp.mean(event_pe_weights, axis=1)

def event_log_covariances(event_pe_weights_n, event_pe_weights_m):
    """
    Compute covariances of log Bayes factors between two sets of event weights.

    Parameters
    ----------
    event_pe_weights_n : jnp.ndarray
        First array of event posterior sample weights, shape (Nobs, NPE).
    event_pe_weights_m : jnp.ndarray
        Second array of event posterior sample weights (same shape as above).

    Returns
    -------
    jnp.ndarray
        Covariances per event, shape (Nobs,).
    """

    assert event_pe_weights_m.shape == event_pe_weights_n.shape
    Nobs, NPE = event_pe_weights_n.shape

    mu_n = reweighted_event_bayes_factors(event_pe_weights_n)
    mu_m = reweighted_event_bayes_factors(event_pe_weights_m)

    cov = jnp.mean(event_pe_weights_n*event_pe_weights_m, axis=1) / mu_n / mu_m - 1
    return cov / (NPE - 1)

def log_likelihood_covariance(vt_weights_n, vt_weights_m, event_pe_weights_n, event_pe_weights_m, total_generated):
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

    event_covs = event_log_covariances(event_pe_weights_n, event_pe_weights_m)
    vt_cov = selection_function_log_covariance(vt_weights_n, vt_weights_m, total_generated)

    return jnp.sum(event_covs) + Nobs**2 * vt_cov

def error_statistics_from_weights(vt_weights, event_weights, total_generated, include_likelihood_correction=True):
    """
    Compute error statistics for hyperposterior, Eqs. 36-39 of ---------------

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
    
    # compute weights according to Eq. 39
    length, Nobs, NPE = event_weights.shape
    axis = jnp.arange(length)
    arr_n, arr_m = jnp.meshgrid(axis, axis, indexing='ij')
    f = lambda n, m: log_likelihood_covariance(vt_weights[n], vt_weights[m], event_weights[n], event_weights[m], total_generated)
    _f = lambda x: f(x, x)
    variances = jax.lax.map(_f, axis)

    @jax_tqdm.scan_tqdm(length, print_rate=1, tqdm_type='std')
    def weight_func(carry, n):
        _f = lambda x: f(arr_n[n,x], arr_m[n,x])
        meanw = jnp.mean(jax.lax.map(_f, axis), axis=0)
        if include_likelihood_correction:
            meanw = likelihood_log_correction(vt_weights[n], total_generated, Nobs) - meanw
        return carry, meanw

    weight_func = jax_tqdm.scan_tqdm(length, print_rate=1, tqdm_type='std')(weight_func)
    _, weights = jax.lax.scan(weight_func, 0., xs=axis)

    precision = float((jnp.mean(variances) - jnp.mean(weights)) / 2 / jnp.log(2))
    accuracy = float(jnp.var(weights) / 2 / jnp.log(2))
    error = float(precision + accuracy)
    
    return {'error_statistic': error, 'precision_statistic': precision, 'accuracy_statistic': accuracy}

def bilby_model_to_model_function(bilby_model, conversion_function=lambda args: (args, None)):
    """
    Wrap a Bilby or gwpopulation jax-compatible model into a callable function interface.

    Parameters
    ----------
    bilby_model : bilby.hyper.model.Model, gwpopulation.experimental.jax.NonCachingModel, or callable
        Model object to be converted. If it is already a callable, it is returned unchanged.
    conversion_function : callable, optional
        Function applied to parameter dictionaries before evaluating the model.
        Should take a dict of parameters and return (modified_parameters, added_keys).

    Returns
    -------
    callable
        A function with signature (data, parameters) -> probability values,
        where `data` is a dictionary of GW parameter samples and `parameters` are
        hyperparameters of the population model.
    """

    from copy import deepcopy
    model_copy = deepcopy(bilby_model)
    if not (isinstance(model_copy, bilby.hyper.model.Model) or isinstance(model_copy, gwpopulation.experimental.jax.NonCachingModel)):
        # TODO: add some catches here, otherwise it assumes a particular form for the model
        return model_copy # function of data, parameters
    

    def model_to_return(data, parameters):
        parameters, added_keys = conversion_function(parameters)
        model_copy.parameters.update(parameters)
        return model_copy.prob(data)
    
    return model_to_return

def _compute_mean_weights_for_correction(hyperposterior, bilby_model, gw_dataset, MC_integral_size=None, conversion_function=lambda args: (args, None), MC_type='single event', verbose=True, rate=False):
    """
    Compute mean event or selection weights integrated over hyperposterior samples. 

    For a Monte Carlo integral
    
    \hat{I}(\Lambda) = \frac{1}{M}\sum_{i=1}^M \frac{p(\theta_i | \Lambda)}{p(\theta_i | {\rm draw})},

    and a set of $N_{\rm samp}$ samples from the hyperposterior $\Lambda_n$, then compute

    \overline{w}_i = \frac{1}{N_{\rm samp}} \sum_{n=1}^{N_{\rm samp}} \frac{1}{\hat{I}(\Lambda_n)}\frac{p(\theta_i | \Lambda_n)}{p(\theta_i | {\rm draw})},

    the weight averaged over the hyperposterior and dividing out $\hat{I}$, for use in computing the error statistics.

    
    Parameters
    ----------
    hyperposterior : pandas.DataFrame
        Hyperposterior samples with columns as model parameter names.
    bilby_model : bilby.hyper.model.Model, gwpopulation.experimental.jax.NonCachingModel, or callable
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

    Returns
    -------
    jnp.ndarray
        Mean normalized event weights across hyperposterior samples,
        shape matching the sampling prior. 
    """

    model_function = bilby_model_to_model_function(bilby_model, conversion_function=conversion_function)

    gw_dataset = gw_dataset.copy()
    sampling_prior = gw_dataset.pop('prior')

    if MC_integral_size is None:
        try:
            MC_integral_size = gw_dataset.pop('total_generated')
        except KeyError:
            MC_integral_size = sampling_prior.shape[-1]

    mean_event_weights = jnp.zeros_like(sampling_prior) # (Nevents, NPE)
    
    n = hyperposterior.shape[0]
    keys = hyperposterior.keys()
    data = jnp.array([hyperposterior[k] for k in keys])
    
    def weights_for_single_sample(ii, mean_event_weights):
        parameters = {k: data[ik,ii] for ik, k in enumerate(keys)}
        weights = model_function(gw_dataset, parameters) / sampling_prior
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

def _compute_integrated_cov(integrated_weights, sample, model_function, gw_dataset, MC_integral_size=None):
    """
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
    expectation = jnp.sum(weights, axis=-1) / MC_integral_size
    var = (-1. + jnp.sum(weights**2, axis=-1) / MC_integral_size / expectation**2) / (MC_integral_size - 1)
    integrated_cov = (-1. + jnp.sum(integrated_weights * weights, axis=-1) / MC_integral_size / expectation) / (MC_integral_size - 1)

    return integrated_cov, var
    
def error_statistics(
        model_function, 
        injections, 
        event_posteriors, 
        hyperposterior, 
        include_likelihood_correction=True, 
        conversion_function=lambda args: (args, None), 
        nobs=None, 
        verbose=True,
        rate=False
        ):
    """
    Compute error, precision, and accuracy statistics from model, hyperposterior, and data.

    Parameters
    ----------
    model_function : bilby.hyper.model.Model, callable
        Population model with interface (dataset, parameters) -> probabilities.
    injections : dict
        Injection dataset, including 'prior' and 'total_generated' keys.
    event_posteriors : dict
        Event posterior samples, including 'prior' key.
    hyperposterior : pandas.DataFrame
        Hyperposterior samples with columns as parameter names.
    include_likelihood_correction : bool, default=True
        Whether to include likelihood correction in accuracy estimate. 
        Set to False if the hyperlikelihood for sampling from the posterior was estimated
        using the unbiased likelihood of Eq. 24 of ----------------
    conversion_function : callable, optional
        Function to convert hyperposterior parameters before model evaluation.
    nobs : int, optional
        Number of observed events. If None, inferred from `event_posteriors`.
    verbose : bool, default=True
        Whether to print progress and summary messages.
    rate : bool, default=False
        Whether to treat the VT weights as rate-weighted. TODO!!!!

    Returns
    -------
    dict
        Dictionary with keys:
        - 'error_statistic' : float, total information loss in bits.
        - 'precision_statistic' : float, information loss due to variance.
        - 'accuracy_statistic' : float, information loss due to bias.
    """

    if nobs is None:
        nobs = event_posteriors['prior'].shape[0]
        if verbose:
            print(f'Nobs not provided, assuming Nobs = {nobs}')
    total_generated = injections['total_generated']
    
    mean_event_weights = _compute_mean_weights_for_correction(
        hyperposterior, 
        model_function, 
        event_posteriors, 
        conversion_function=conversion_function, 
        MC_type='single event', 
        verbose=verbose
        )
    mean_vt_weights = _compute_mean_weights_for_correction(
        hyperposterior, 
        model_function, 
        injections, 
        MC_integral_size=total_generated, 
        conversion_function=conversion_function, 
        MC_type='selection', 
        verbose=verbose,
        rate=rate
        )
    
    n = hyperposterior.shape[0]

    hyperposterior_samples = hyperposterior.to_dict(orient='list')
    for k in hyperposterior_samples.keys():
        hyperposterior_samples[k] = jnp.array(hyperposterior_samples[k])

    def create_loop_fn(m, p, MC_type='single event'):
        _model_function = bilby_model_to_model_function(model_function, conversion_function=conversion_function)
        loop_fn = lambda _, sample: (_, (sample[0],)+ _compute_integrated_cov(
                m,
                sample[1], 
                _model_function, 
                p,
                ))
        if verbose:
            return jax_tqdm.scan_tqdm(n, print_rate=1, tqdm_type='std', desc=f'For each posterior sample, average {MC_type} covariance with another posterior sample')(loop_fn)
        else:
            return jax.jit(loop_fn)

    _, (_, event_integrated_covs, event_vars) = jax.lax.scan(
        create_loop_fn(mean_event_weights, event_posteriors),
        0,
        (jnp.arange(n), hyperposterior_samples),
        length=n
    )
    
    _, (_, vt_integrated_covs, vt_vars) = jax.lax.scan(
        create_loop_fn(mean_vt_weights, injections, MC_type='selection'),
        0,
        (jnp.arange(n), hyperposterior_samples),
        length=n
    )

    var = jnp.sum(event_vars, axis=-1) + nobs**2 * vt_vars
    cov = jnp.sum(event_integrated_covs, axis=-1) + nobs**2 * vt_integrated_covs

    event_precision = float(jnp.mean(jnp.sum(event_vars - event_integrated_covs, axis=-1)) / 2 / jnp.log(2))
    vt_precision = float(nobs**2 * jnp.mean(vt_vars - vt_integrated_covs) / 2 / jnp.log(2))
    
    precision = float((jnp.mean(var) - jnp.mean(cov)) / 2 / jnp.log(2))
    if include_likelihood_correction:
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
