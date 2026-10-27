"""Merton (1976) and Kou (2002) jump-diffusion models for daily log returns.

Densities, simulators, threshold starting values and maximum-likelihood fitting,
plus geometric Brownian motion and GARCH(1,1) as benchmarks.
Parameters are annualised; dt is the period length in years (1/252 for daily data).
Python port (and extension) of the MATLAB functions in matlab/.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln
from scipy.stats import norm, poisson

MERTON_PARAMS = ["mu", "sigma", "lambda", "mu_j", "sigma_j"]
KOU_PARAMS = ["mu", "sigma", "lambda", "p", "eta1", "eta2"]
GBM_PARAMS = ["mu", "sigma"]
GARCH_PARAMS = ["c", "omega", "alpha", "beta"]


# ---------------------------------------------------------------- densities

def gbm_pdf(r, dt, mu, sigma):
    """Normal density of log returns under geometric Brownian motion."""
    return norm.pdf(r, (mu - sigma**2 / 2) * dt, sigma * np.sqrt(dt))


def merton_pdf(r, dt, mu, sigma, lam, mu_j, sigma_j, nterm=None):
    """Poisson mixture of normals: k jumps give mean m + k*mu_j, variance sigma^2*dt + k*sigma_j^2.

    The sum is truncated where the remaining Poisson probability is below 1e-12.
    """
    m = (mu - sigma**2 / 2) * dt
    if lam <= 0:
        return norm.pdf(r, m, sigma * np.sqrt(dt))
    if nterm is None:
        nterm = int(poisson.ppf(1 - 1e-12, lam * dt)) + 2
    k = np.arange(nterm + 1)
    logw = k * np.log(lam * dt) - gammaln(k + 1) - lam * dt          # Poisson weights
    sd = np.sqrt(sigma**2 * dt + sigma_j**2 * k)
    dens = norm.pdf(np.asarray(r)[:, None], m + mu_j * k, sd)
    return dens @ np.exp(logw)


def kou_pdf(r, dt, mu, sigma, lam, p, eta1, eta2, half_width=0.75, n=2**15):
    """Exact Kou density, by Fourier inversion of the characteristic function.

    Valid for any number of jumps per period:
    phi(u) = exp(i*u*m - sigma^2*dt*u^2/2 + lam*dt*(p*eta1/(eta1 - i*u) + q*eta2/(eta2 + i*u) - 1)).
    The density is computed on a grid of n points over [-half_width, half_width) and interpolated.
    """
    m = (mu - sigma**2 / 2) * dt
    dx = 2 * half_width / n
    k = np.arange(n)
    u = 2 * np.pi * (k - n / 2) / (n * dx)
    jump_cf = p * eta1 / (eta1 - 1j * u) + (1 - p) * eta2 / (eta2 + 1j * u)
    phi = np.exp(1j * u * m - 0.5 * sigma**2 * dt * u**2 + lam * dt * (jump_cf - 1))
    # f(x_j) = (du/2pi) sum_k phi(u_k) exp(-i u_k x_j), with x_j = -half_width + j*dx
    f = np.fft.fft(phi * np.exp(1j * u * half_width)) * (-1.0) ** k
    f = np.real(f) * (2 * np.pi / (n * dx)) / (2 * np.pi)
    x = -half_width + k * dx
    return np.interp(r, x, np.maximum(f, 1e-300))


def kou_pdf_approx(r, dt, mu, sigma, lam, p, eta1, eta2):
    """Kou's small-dt approximation: at most one double-exponential jump per period.

    Accurate only when lam*dt is small; kept for comparison with the MATLAB code.
    """
    x = np.asarray(r) - (mu - sigma**2 / 2) * dt
    sd = sigma * np.sqrt(dt)
    diffusion = norm.pdf(x, 0, sd)
    if lam <= 0:
        return diffusion
    # log-space products avoid overflow of exp(-x*eta) for large |x|
    up = p * eta1 * np.exp(0.5 * sigma**2 * eta1**2 * dt - x * eta1
                           + norm.logcdf((x - eta1 * sigma**2 * dt) / sd))
    down = (1 - p) * eta2 * np.exp(0.5 * sigma**2 * eta2**2 * dt + x * eta2
                                   + norm.logcdf(-(x + eta2 * sigma**2 * dt) / sd))
    return (1 - lam * dt) * diffusion + lam * dt * (up + down)


def garch_variance(R, c, omega, alpha, beta):
    """Conditional variances of a GARCH(1,1): s2_t = omega + alpha*e_{t-1}^2 + beta*s2_{t-1}."""
    e2 = (R - c) ** 2
    s2_0 = np.var(R)                     # first variance set to the sample variance
    const = omega / (1 - beta)
    y = lfilter([alpha], [1, -beta], e2[:-1], zi=[beta * (s2_0 - const)])[0]
    return np.concatenate([[s2_0], y + const])


def garch_logpdf(R, c, omega, alpha, beta):
    """Log densities of daily returns under GARCH(1,1) with normal shocks (parameters in daily units)."""
    s2 = garch_variance(R, c, omega, alpha, beta)
    return norm.logpdf(R, c, np.sqrt(s2))


def garch_jump_logpdf(R, c, omega, alpha, beta, lam_d, mu_j, sigma_j):
    """GARCH(1,1) diffusion plus Merton-type jumps (parameters in daily units).

    r_t = c + sqrt(h_t) z_t + sum of N_t normal jumps, N_t ~ Poisson(lam_d).
    h_t follows a GARCH(1,1) driven by the total demeaned shock, so volatility
    clustering and jumps are estimated together.
    """
    h = garch_variance(R, c + lam_d * mu_j, omega, alpha, beta)
    nterm = int(poisson.ppf(1 - 1e-12, lam_d)) + 2
    k = np.arange(nterm + 1)
    w = np.exp(k * np.log(lam_d) - gammaln(k + 1) - lam_d) if lam_d > 0 else (k == 0).astype(float)
    dens = norm.pdf(R[:, None], c + mu_j * k, np.sqrt(h[:, None] + sigma_j**2 * k))
    return np.log(dens @ w)


# --------------------------------------------------------------- simulators

def merton_rvs(n, dt, mu, sigma, lam, mu_j, sigma_j, rng):
    """Simulate n log returns under the Merton model."""
    dN = rng.poisson(lam * dt, n)
    Y = mu_j * dN + sigma_j * np.sqrt(dN) * rng.standard_normal(n)
    return (mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n) + Y


def garch_jump_rvs(n, c, omega, alpha, beta, lam_d, mu_j, sigma_j, rng, burn=500):
    """Simulate n daily returns from GARCH(1,1) plus normal jumps (lam_d = 0 gives plain GARCH)."""
    N = n + burn
    dN = rng.poisson(lam_d, N)
    J = mu_j * dN + sigma_j * np.sqrt(dN) * rng.standard_normal(N)
    z = rng.standard_normal(N)
    r = np.empty(N)
    h = omega / (1 - alpha - beta)
    for t in range(N):
        r[t] = c + np.sqrt(h) * z[t] + J[t]
        h = omega + alpha * (r[t] - c - lam_d * mu_j) ** 2 + beta * h
    return r[burn:]


def kou_rvs(n, dt, mu, sigma, lam, p, eta1, eta2, rng):
    """Simulate n log returns under the Kou model (double-exponential jumps)."""
    dN = rng.poisson(lam * dt, n)
    Y = np.zeros(n)
    for k in range(1, dN.max(initial=0) + 1):
        idx = dN >= k
        m = idx.sum()
        isup = rng.random(m) < p
        Y[idx] += np.where(isup, rng.exponential(1 / eta1, m), -rng.exponential(1 / eta2, m))
    return (mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n) + Y


# ---------------------------------------------------------- starting values

def merton_start(R, dt, epsilon=0.02):
    """Treat |returns| > epsilon as jumps and use simple moments of each group."""
    jump = np.abs(R) > epsilon
    lam = jump.sum() / (len(R) * dt)
    Rd, Rj = R[~jump], R[jump]
    sigma = Rd.std(ddof=1) / np.sqrt(dt)
    mu = (2 * Rd.mean() + sigma**2 * dt) / (2 * dt)
    sigma_j = np.sqrt(max(Rj.var(ddof=1) - sigma**2 * dt, 1e-8))
    mu_j = Rj.mean() - (mu - sigma**2 / 2) * dt
    return np.array([mu, sigma, lam, mu_j, sigma_j])


def kou_start(R, dt, epsilon=0.02):
    up, down = R > epsilon, R < -epsilon
    lam = (up.sum() + down.sum()) / (len(R) * dt)
    Rd = R[np.abs(R) <= epsilon]
    sigma = Rd.std(ddof=1) / np.sqrt(dt)
    mu = (2 * Rd.mean() + sigma**2 * dt) / (2 * dt)
    p = up.sum() / (up.sum() + down.sum())
    return np.array([mu, sigma, lam, p, 1 / R[up].mean(), -1 / R[down].mean()])


def gbm_start(R, dt):
    sigma = R.std(ddof=1) / np.sqrt(dt)
    return np.array([(R.mean() + sigma**2 * dt / 2) / dt, sigma])


def garch_start(R, dt):
    return np.array([R.mean(), 0.02 * R.var(), 0.08, 0.90])


def garch_jump_start(R, dt):
    return np.array([R.mean(), 0.02 * R.var(), 0.06, 0.90, 0.05, -0.005, 2 * R.std()])


# ------------------------------------------------------------------ fitting

def _valid(model, th):
    if model == "merton":
        return th[1] > 0 and th[2] >= 0 and th[4] > 0
    if model == "kou":   # eta1 > 1 keeps E[exp(Y)] finite
        return th[1] > 0 and th[2] >= 0 and 0 < th[3] < 1 and th[4] > 1 and th[5] > 0
    if model == "garch":
        return th[1] > 0 and th[2] >= 0 and th[3] >= 0 and th[2] + th[3] < 1
    if model == "garch_jump":
        return (th[1] > 0 and th[2] >= 0 and th[3] >= 0 and th[2] + th[3] < 1
                and 0 <= th[4] < 1 and th[6] > 0)
    return th[1] > 0


PDF = {"gbm": gbm_pdf, "merton": merton_pdf, "kou": kou_pdf}
LOGPDF = {"garch": garch_logpdf, "garch_jump": garch_jump_logpdf}
START = {"gbm": gbm_start, "merton": merton_start, "kou": kou_start, "garch": garch_start,
         "garch_jump": garch_jump_start}
NAMES = {"gbm": GBM_PARAMS, "merton": MERTON_PARAMS, "kou": KOU_PARAMS, "garch": GARCH_PARAMS,
         "garch_jump": GARCH_PARAMS + ["lambda_daily", "mu_j", "sigma_j"]}


def nll(th, R, dt, model):
    """Negative log-likelihood; infinite outside the valid parameter region."""
    if not _valid(model, th):
        return np.inf
    if model in LOGPDF:
        v = -LOGPDF[model](R, *th).sum()
        return v if np.isfinite(v) else np.inf
    d = PDF[model](R, dt, *th)
    if np.any(~np.isfinite(d)) or np.any(d <= 0):
        return np.inf
    return -np.log(d).sum()


def fit(R, dt, model, theta0=None, restarts=3):
    """Maximum likelihood by Nelder-Mead (as MATLAB's fminsearch), restarted from its own optimum."""
    th = START[model](R, dt) if theta0 is None else np.asarray(theta0, float)
    for _ in range(restarts):
        res = minimize(nll, th, args=(R, dt, model), method="Nelder-Mead",
                       options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-8, "fatol": 1e-8})
        th = res.x
    return th, res.fun
