"""Compare jump-diffusion, GARCH and GARCH-with-jumps models across stock markets.

Usage:
    python get_prices.py ^GSPC ^N225 ^FTSE ^GDAXI ^AXJO ^BVSP ^MXX ^BSESN 000001.SS ^J200.JO ^JKSE \
        --start 2005-01-01 --end 2026-01-01
    python run_analysis.py            # all markets in MARKETS
    python run_analysis.py GSPC N225  # a subset

For each market the script fits five models to daily log returns by maximum likelihood:
  gbm         geometric Brownian motion (normal returns)
  merton      Merton (1976) jump-diffusion, normal jumps
  kou         Kou (2002) jump-diffusion, double-exponential jumps (exact density)
  garch       GARCH(1,1) with normal shocks (volatility clustering, no jumps)
  garch_jump  GARCH(1,1) plus normal jumps (clustering and jumps together)
It then re-estimates the Kou and GARCH-jump models on a path simulated from their own fit (parameter
recovery), and compares tails: kurtosis, frequency of large falls and the 1% daily Value at Risk.
Results go to results/ and figures/.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, kurtosis, norm, skew

import jump_diffusion as jd

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 1 / 252
NSIM = 100          # simulated paths pooled for model-implied tail measures
MODELS = ["gbm", "merton", "kou", "garch", "garch_jump"]

# file name: (market, index, group)
MARKETS = {
    "GSPC": ("United States", "S&P 500", "Developed"),
    "N225": ("Japan", "Nikkei 225", "Developed"),
    "FTSE": ("United Kingdom", "FTSE 100", "Developed"),
    "GDAXI": ("Germany", "DAX", "Developed"),
    "AXJO": ("Australia", "S&P/ASX 200", "Developed"),
    "BVSP": ("Brazil", "Ibovespa", "Emerging"),
    "MXX": ("Mexico", "IPC", "Emerging"),
    "BSESN": ("India", "BSE Sensex", "Emerging"),
    "000001.SS": ("China", "SSE Composite", "Emerging"),
    "J200.JO": ("South Africa", "FTSE/JSE Top 40", "Emerging"),
    "JKSE": ("Indonesia", "Jakarta Composite", "Emerging"),
}


def load_returns(name):
    s = pd.read_csv(os.path.join(HERE, "data", name + ".csv"), index_col=0, parse_dates=True)["close"]
    return s, np.diff(np.log(s.values))


def simulate(model, n, th, rng):
    if model == "gbm":
        return rng.normal((th[0] - th[1]**2 / 2) * DT, th[1] * np.sqrt(DT), n)
    if model == "merton":
        return jd.merton_rvs(n, DT, *th, rng)
    if model == "kou":
        return jd.kou_rvs(n, DT, *th, rng)
    if model == "garch":
        return jd.garch_jump_rvs(n, *th, 0.0, 0.0, 1e-9, rng)
    return jd.garch_jump_rvs(n, *th, rng)


def tail_stats(x):
    return {"excess kurtosis": kurtosis(x), "skewness": skew(x),
            "falls > 5% per year": 252 * np.mean(x < -0.05),
            "1% VaR": -np.quantile(x, 0.01)}


def analyse(name, rng):
    market, index, group = MARKETS[name]
    prices, R = load_returns(name)
    n = len(R)
    base = {"market": market, "index": index, "group": group}
    fits, comp, params = {}, [], []
    for model in MODELS:
        th, f = jd.fit(R, DT, model)
        fits[model] = th
        k = len(th)
        comp.append({**base, "model": model, "params": k, "loglik": -f,
                     "AIC": 2 * k + 2 * f, "BIC": k * np.log(n) + 2 * f})
        rec = [np.nan] * k
        if model in ("kou", "garch_jump"):     # parameter recovery on one simulated path
            rec, _ = jd.fit(simulate(model, n, th, rng), DT, model, theta0=th)
        for pname, est, r in zip(jd.NAMES[model], th, rec):
            params.append({**base, "model": model, "parameter": pname, "estimate": est,
                           "re-estimate on simulated path": r})
    comp = pd.DataFrame(comp)
    ll = comp.set_index("model")["loglik"]

    # tails: empirical against pooled simulations from each fitted model
    tails = [{**base, "series": "empirical", **tail_stats(R)}]
    for model in ["gbm", "kou", "garch_jump"]:
        tails.append({**base, "series": model, **tail_stats(simulate(model, n * NSIM, fits[model], rng))})

    # headline summary
    mu, sigma, lam, p, eta1, eta2 = fits["kou"]
    kou_jump_var = lam * (2 * p / eta1**2 + 2 * (1 - p) / eta2**2)
    c, omega, alpha, beta, lam_d, mu_j, sigma_j = fits["garch_jump"]
    gj_jump_var = lam_d * (mu_j**2 + sigma_j**2)
    gj_diff_var = (omega + alpha * gj_jump_var) / (1 - alpha - beta)   # E[h]: jumps also feed the GARCH
    p_big = norm.cdf(-0.03, mu_j, sigma_j) + norm.sf(0.03, mu_j, sigma_j)
    summary = {
        **base, "start": prices.index[0].date(), "end": prices.index[-1].date(), "days": n,
        "annual volatility": R.std(ddof=1) * np.sqrt(252),
        "excess kurtosis": kurtosis(R), "skewness": skew(R),
        "1% VaR empirical": -np.quantile(R, 0.01),
        "1% VaR normal": -norm.ppf(0.01, R.mean(), R.std(ddof=1)),
        "loglik gain kou vs gbm": ll["kou"] - ll["gbm"],
        "loglik gain garch vs gbm": ll["garch"] - ll["gbm"],
        "loglik gain garch_jump vs garch": ll["garch_jump"] - ll["garch"],
        "kou jumps per year": lam,
        "kou share of jumps down": 1 - p,
        "kou mean up jump": 1 / eta1, "kou mean down jump": 1 / eta2,
        "kou jump share of variance": kou_jump_var / (sigma**2 + kou_jump_var),
        "garch persistence": alpha + beta,
        "gj jumps per year": 252 * lam_d,
        "gj mean jump": mu_j, "gj jump sd": sigma_j,
        "gj jumps over 3% per year": 252 * lam_d * p_big,
        "gj jump share of variance": gj_jump_var / (gj_jump_var + gj_diff_var),
    }
    return comp, pd.DataFrame(params), pd.DataFrame(tails), summary, (prices, R, fits)


def plot_densities(curves):
    """Small multiples: empirical density against GBM and Kou, log scale."""
    names = list(curves)
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.7 * rows), sharex=True, sharey=True)
    x = np.linspace(-0.1, 0.1, 500)
    for ax, name in zip(axes.flat, names):
        prices, R, fits = curves[name]
        ax.semilogy(x, gaussian_kde(R, bw_method=0.15)(x), color="#222222", lw=2, label="Data")
        ax.semilogy(x, jd.gbm_pdf(x, DT, *fits["gbm"]), color="#9aa0a6", ls="--", lw=1.4, label="Normal (GBM)")
        ax.semilogy(x, jd.kou_pdf(x, DT, *fits["kou"]), color="#d1242f", lw=1.4, label="Kou jump-diffusion")
        market, _, group = MARKETS[name]
        ax.set_title(f"{market} ({group.lower()})", fontsize=10)
        ax.set_ylim(5e-2, 80)
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    spare = axes.flat[len(names)] if len(names) < rows * cols else axes.flat[0]
    spare.legend(handles, labels, frameon=False, fontsize=10, loc="center")
    fig.supxlabel("Daily log return")
    fig.supylabel("Density (log scale)")
    fig.suptitle("Daily index returns 2005-2025: the normal model misses the tails", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "densities.png"), dpi=130)
    plt.close(fig)


def main(names):
    rng = np.random.default_rng(2026)
    out, curves = [], {}
    for name in names:
        print("fitting", name, flush=True)
        comp, params, tails, summary, curve = analyse(name, rng)
        out.append((comp, params, tails, summary))
        curves[name] = curve
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    tables = {"model_comparison": pd.concat([o[0] for o in out]),
              "parameters": pd.concat([o[1] for o in out]),
              "tails": pd.concat([o[2] for o in out]),
              "summary": pd.DataFrame([o[3] for o in out])}
    for label, df in tables.items():
        df.to_csv(os.path.join(HERE, "results", label + ".csv"), index=False, float_format="%.6g")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(tables["summary"].T.to_string())
    plot_densities(curves)


def replot():
    """Redraw figures from results/parameters.csv without refitting."""
    p = pd.read_csv(os.path.join(HERE, "results", "parameters.csv"))
    curves = {}
    for name, (market, _, _) in MARKETS.items():
        prices, R = load_returns(name)
        q = p[p["market"] == market]
        fits = {m: q[q["model"] == m]["estimate"].values for m in ["gbm", "kou"]}
        curves[name] = (prices, R, fits)
    plot_densities(curves)


if __name__ == "__main__":
    if sys.argv[1:] == ["--replot"]:
        replot()
    else:
        main(sys.argv[1:] or list(MARKETS))
