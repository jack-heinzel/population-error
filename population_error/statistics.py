import jax.numpy as jnp
import jax
import jax_tqdm

def selection_function(weights, total_samples):
    """
    Compute the selection function given weights and total number of injections.

    Parameters
    ----------
    weights : jnp.ndarray
        Array of importance weights for injection samples.
    total_samples : int or float
        Total number of injections.

    Returns
    -------
    float
        Estimated selection function value (mean weight normalized by total samples).
    """
    return jnp.sum(weights) / total_samples

def selection_function_log_covariance(weights_n, weights_m, total_samples):
    """
    Compute the covariance of log selection functions between two weight sets.

    Parameters
    ----------
    weights_n : jnp.ndarray
        First set of importance weights.
    weights_m : jnp.ndarray
        Second set of importance weights (must match shape of weights_n).
    total_samples : int or float
        Total number of injections.

    Returns
    -------
    float
        Covariance between log selection function estimates.
    """
    assert weights_n.shape == weights_m.shape
    mu_n, mu_m = selection_function(weights_n, total_samples), selection_function(weights_m, total_samples)
    cov = jnp.sum(weights_n * weights_m) / total_samples / mu_n / mu_m - 1
    return cov / (total_samples-1)

def likelihood_log_correction(weights, total_samples, Nobs):
    """
    Compute the likelihood log-correction term for variance estimation.

    Parameters
    ----------
    weights : jnp.ndarray
        Importance weights for injection samples.
    total_samples : int or float
        Total number of injections.
    Nobs : int
        Number of observed events.

    Returns
    -------
    float
        Likelihood log-correction value.
    """
    var = selection_function_log_covariance(weights, weights, total_samples)
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

def log_likelihood_covariance(vt_weights_n, vt_weights_m, event_pe_weights_n, event_pe_weights_m, total_samples):
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
    total_samples : int or float
        Total number of injections.

    Returns
    -------
    float
        Log-likelihood covariance estimate.
    """

    Nobs, NPE = event_pe_weights_n.shape

    event_covs = event_log_covariances(event_pe_weights_n, event_pe_weights_m)
    vt_cov = selection_function_log_covariance(vt_weights_n, vt_weights_m, total_samples)

    return jnp.sum(event_covs) + Nobs**2 * vt_cov

def error_statistics(vt_weights, event_weights, total_samples, include_likelihood_correction=True):
    """
    Compute error statistics for hyperposterior, Eqs. 36-39 of ---------------

    Parameters
    ----------
    vt_weights : jnp.ndarray
        Array of shape (n_samples, n_injections), injection weights per hyperposterior sample.
    event_weights : jnp.ndarray
        Array of shape (n_samples, n_obs, n_pe), event posterior weights per hyperposterior sample.
    total_samples : int or float
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
    f = lambda n, m: log_likelihood_covariance(vt_weights[n], vt_weights[m], event_weights[n], event_weights[m], total_samples)
    _f = lambda x: f(x, x)
    variances = jax.lax.map(_f, axis)

    @jax_tqdm.scan_tqdm(length, print_rate=1, tqdm_type='std')
    def weight_func(carry, n):
        _f = lambda x: f(arr_n[n,x], arr_m[n,x])
        meanw = jnp.mean(jax.lax.map(_f, axis), axis=0)
        if include_likelihood_correction:
            meanw = likelihood_log_correction(vt_weights[n], total_samples, Nobs) - meanw
        return carry, meanw

    weight_func = jax_tqdm.scan_tqdm(length, print_rate=1, tqdm_type='std')(weight_func)
    _, weights = jax.lax.scan(weight_func, 0., xs=axis)

    precision = (jnp.mean(variances) - jnp.mean(weights)) / 2 / jnp.log(2)
    accuracy = jnp.var(weights) / 2 / jnp.log(2)
    error = precision + accuracy
    
    return precision, accuracy, error 
