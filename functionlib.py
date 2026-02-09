"""
This file contains the functions required for Data-Driven Bayesian Empirical Fragility Modelling.

Developed by UCL RDR Team: Hossein Ebrahimian, Jingren Wu, Fatemeh Jalayer

Current version: v0 - Fixed IM, Single Class, Uncertain Model parameters

Last modified: Jan 2026
"""

import numpy as np
import statsmodels.api as sm
from scipy.special import gamma
from scipy.stats import norm, beta as beta_dist


# --------------------------------------------------
def glm_hierarchical(DS, NDS, GRM):

    """
    DS  : list of arrays 
    NDS : number of damage states (int)
    GRM : link function name ('logit', 'probit', 'cloglog', etc.)
    """
    GRMCoef = np.zeros((2, NDS))

    # Choose link function
    if GRM.lower() == 'logit':
        link = sm.families.links.Logit()
    elif GRM.lower() == 'probit':
        link = sm.families.links.probit()
    elif GRM.lower() in ['cloglog', 'comploglog']:
        link = sm.families.links.cloglog()
    else:
        raise ValueError(f"Unsupported link function: {GRM}")

    for j in range(NDS):
        DSj = DS[j].copy()
        num_collapse = np.zeros(len(DS[j]))

        # Append higher damage states 
        for k in range(j + 1, NDS + 1):
            DSj = np.concatenate((DSj, DS[k]))
            num_collapse = np.concatenate((num_collapse, np.ones(len(DS[k]))))

        # Adding intercept
        X = np.log(DSj)
        X = sm.add_constant(X)  # adds intercept column

        model = sm.GLM(num_collapse, X,
                       family=sm.families.Binomial(link=link))
        results = model.fit()

        GRMCoef[:, j] = results.params  # [intercept, slope]

    # MATLAB reshape(GRMCoef,2*NDS,1) is column-wise flatten
    THETA = GRMCoef.reshape((2 * NDS, 1), order='F')

    return THETA

# --------------------------------------------------

def likelihoodFunction_DS(dataDS, theta, GRM):
    
    """
    dataDS : list of arrays 
    theta  : 1D numpy array of parameters
    GRM    : string -> 'logit', 'probit', or 'comploglog'
    """

    # ---- Define Plink ----
    if GRM == 'logit':
        def Plink(X, bo, b1):
            return 1 - 1 / (1 + np.exp(-bo - b1 * np.log(X)))

    elif GRM == 'probit':
        def Plink(X, bo, b1):
            return 1 - norm.cdf(bo + b1 * np.log(X))

    elif GRM == 'comploglog':
        def Plink(X, bo, b1):
            return np.exp(-np.exp(bo + b1 * np.log(X)))

    else:
        raise ValueError("GRM must be 'logit', 'probit', or 'comploglog'")

    # ---- Calculate P(DS | IM) ----
    NDS = len(dataDS) - 1
    PDS_IM = [None] * (NDS + 1)

    for j in range(NDS + 1):
        X = np.asarray(dataDS[j])
        n = len(X)

        Pl = np.zeros((n, j + 1))

        # First damage state
        Pl[:, 0] = Plink(X, theta[0], theta[1])

        if j < NDS:
            for k in range(1, j + 1):
                Pl[:, k] = (
                    Plink(X, theta[2*k], theta[2*k + 1])
                    * (1 - np.sum(Pl[:, :k], axis=1))
                )
            PDS_IM[j] = Pl[:, j]

        else:  # last state
            for k in range(1, NDS):
                Pl[:, k] = (
                    Plink(X, theta[2*k], theta[2*k + 1])
                    * (1 - np.sum(Pl[:, :k], axis=1))
                )
            PDS_IM[j] = 1 - np.sum(Pl[:, :NDS], axis=1)

    # ---- Log-likelihood ----
    lny = 0.0
    for arr in PDS_IM:
        positive_vals = arr[arr > 0]
        lny += np.sum(np.log(positive_vals))

    return lny


# --------------------------------------------------
def randmvn(mu, sigma):
    
    """
    Sample from a multivariate normal using Cholesky decomposition.
    """
    mu = np.atleast_1d(mu)
    sigma = np.atleast_2d(sigma)

    # Cholesky decomposition (lower-triangular)
    L = np.linalg.cholesky(sigma)

    # Standard normal vector
    z = np.random.randn(mu.shape[0])

    # Sample
    sample_theta = mu + L @ z

    return sample_theta

# --------------------------------------------------
def postMCMC(theta, theta_MLE, sigma_theta, loglikelihoodfunc, data):
    
    """
    First Chain Metropolis-Hastings MCMC update.
    """
    DS  = data[0]
    GRM = data[1]
    
    # Proposal
    cov_proposal = np.diag(sigma_theta ** 2)

    # Sample new theta from multivariate normal
    new_theta = randmvn(theta_MLE, cov_proposal)

    # Compute log-likelihoods
    logLikelihoodnew = loglikelihoodfunc(DS, new_theta, GRM)
    logLikelihood = loglikelihoodfunc(DS, theta, GRM)

    # Metropolis-Hastings acceptance probability
    logpratio = logLikelihoodnew - logLikelihood

    u = np.random.rand()
    if np.log(u) < logpratio:
        sample_theta = new_theta
    else:
        sample_theta = theta

    return sample_theta

# --------------------------------------------------
def randAGRW(mu, seeds):
    
    """
    Sample a new theta using Gaussian random walk with scaled covariance.
    """
    mu = np.atleast_1d(mu)
    d = len(mu)
    
    S = (2.38 ** 2) / d * np.cov(seeds, rowvar=True)
    reg = 1e-13 * np.eye(d)
    
    # Cholesky decomposition (lower-triangular)
    while True:
        try:
            L = np.linalg.cholesky(S)
            break
        except np.linalg.LinAlgError:
            S += reg  # add tiny diagonal
    
    # Sample from multivariate normal
    z = np.random.randn(d)
    sample_mu = mu + L @ z
    
    return sample_mu

# --------------------------------------------------
def postMCMCupdated(theta, seeds, loglikelihoodfunc, data):
    
    """
    Single-step Metropolis-Hastings MCMC update using Gaussian random walk.
    """  
    DS  = data[0]
    GRM = data[1]
    
    # Sample a new THETA
    new_theta = randAGRW(theta, seeds)  # previously sampleTheta_GaussianRandomWalk
       
    # Evaluate log-likelihoods
    logLikelihoodnew = loglikelihoodfunc(DS, new_theta, GRM)
    logLikelihood = loglikelihoodfunc(DS, theta, GRM)
    
    # Compute Metropolis-Hastings acceptance probability
    logpratio = logLikelihoodnew - logLikelihood
    
    u = np.random.rand()
    if np.log(u) < logpratio:
        sample_theta = new_theta
    else:
        sample_theta = theta

    return sample_theta

# --------------------------------------------------
def reliability(data):
    """
    Calculate the mean and standard deviation (sigma) of samples.
    """
    firstM = np.mean(data, axis=1)
    secondM = np.mean(data**2, axis=1)
    sigma = np.sqrt(np.real(secondM - firstM**2))
    
    return firstM, sigma

# --------------------------------------------------
def calculate_fragility(IM, theta, NDS, GRM):
    """
    IM   : (n,) array of intensity measures
    theta: parameter vector
    NDS  : number of damage states
    GRM  : 'logit', 'probit', or 'comploglog'
    """

    IM = np.asarray(IM)
    n = len(IM)

    # ---- Define 1 - pi (Plink) ----
    if GRM == 'logit':
        def Plink(X, bo, b1):
            return 1 - 1 / (1 + np.exp(-bo - b1 * np.log(X)))

    elif GRM == 'probit':
        def Plink(X, bo, b1):
            return 1 - norm.cdf(bo + b1 * np.log(X))

    elif GRM == 'comploglog':
        def Plink(X, bo, b1):
            return np.exp(-np.exp(bo + b1 * np.log(X)))

    else:
        raise ValueError("GRM must be 'logit', 'probit', or 'comploglog'")

    # ---- Calculate P(DS | IM) ----
    PDS_IM = [None] * (NDS + 1)
    fragility = np.zeros((n, NDS))

    for j in range(NDS + 1):
        Pl = np.zeros((n, j + 1))

        Pl[:, 0] = Plink(IM, theta[0], theta[1])

        if j < NDS:
            for k in range(1, j + 1):
                Pl[:, k] = (
                    Plink(IM, theta[2*k], theta[2*k + 1]) *
                    (1 - np.sum(Pl[:, :k], axis=1))
                )
            PDS_IM[j] = Pl[:, j]
        else:
            for k in range(1, NDS):
                Pl[:, k] = (
                    Plink(IM, theta[2*k], theta[2*k + 1]) *
                    (1 - np.sum(Pl[:, :k], axis=1))
                )
            PDS_IM[j] = 1 - np.sum(Pl[:, :NDS], axis=1)

    # ---- Calculate fragility curves ----
    sumP = np.zeros(n)
    for j in range(NDS, 0, -1):
        sumP += PDS_IM[j]
        fragility[:, j-1] = sumP

    return fragility, PDS_IM

# --------------------------------------------------







