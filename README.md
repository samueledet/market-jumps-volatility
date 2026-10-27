# Jump-Diffusion Models of Stock Returns: Merton and Kou

MATLAB code to fit two classic jump-diffusion models to daily stock-index returns by maximum likelihood, simulate
from the fitted models and test how well they reproduce the empirical return distribution.

Daily stock returns have fatter tails and sharper peaks than the normal distribution that geometric Brownian motion
(the Black–Scholes model) implies. Jump-diffusion models add occasional jumps to the Brownian motion, which
captures crashes and sudden rallies. This matters for risk measurement (Value at Risk, expected shortfall) and for
pricing options whose value depends on the tails.

## The models

Both models write the log return over a period of length Δt as

```
r = (μ − σ²/2)Δt + σ√Δt·Z + (sum of jumps in the period)
```

where *Z* is standard normal and jumps arrive as a Poisson process with intensity λ per year.

| | Merton (1976) | Kou (2002) |
|---|---|---|
| Jump sizes | Normal, mean μⱼ and std. dev. σⱼ | Double-exponential: up jumps Exp(η₁) with probability *p*, down jumps Exp(η₂) with probability 1 − *p* |
| Parameters | μ, σ, λ, μⱼ, σⱼ | μ, σ, λ, *p*, η₁, η₂ |
| Density used | Poisson mixture of normals (truncated at 100 jumps) | Kou's small-Δt approximation (at most one jump per day) |
| Main feature | Symmetric-shaped jumps | Asymmetric up and down jumps with heavier tails; closed-form prices for many path-dependent options |

## Contents

| File | What it does |
|---|---|
| [`fit_merton.m`](fit_merton.m) | Fits the Merton model and runs the checks below |
| [`fit_kou.m`](fit_kou.m) | Fits the Kou model and runs the checks below |
| [`logmertonpdf.m`](logmertonpdf.m), [`logkoupdf.m`](logkoupdf.m) | Density of daily log returns under each model |
| [`logmertonrnd.m`](logmertonrnd.m), [`logkournd.m`](logkournd.m) | Simulate log-return paths from each model |
| [`load_prices.m`](load_prices.m) | Reads a CSV of daily closing prices |

## What the fitting scripts do

1. **Starting values.** Daily returns larger than 2% in absolute value are treated as jumps and the rest as
   diffusion. Simple moments of the two groups give starting values for every parameter.
2. **Maximum likelihood.** `fminsearch` minimises the negative log-likelihood. Parameter values outside the valid
   region (for example a negative volatility, or *p* outside 0 to 1) get an infinite penalty.
3. **Parameter recovery check.** A path is simulated from the fitted model and the model is re-estimated on it.
   Estimates close to the fitted values show that the parameters are identified.
4. **Goodness of fit.** A two-sample Kolmogorov–Smirnov test compares empirical and simulated returns, and the
   two kernel densities are plotted together.

## Running the code

Requirements: MATLAB R2016b or later (the scripts use local functions) and the Statistics and Machine Learning
Toolbox (`normpdf`, `normcdf`, `poissrnd`, `exprnd`, `kstest2`, `ksdensity`).

1. Put a CSV of daily closing prices in the folder. Either a single column of prices or a file whose last column is
   the price works. For example, the Nikkei 225 series can be downloaded from FRED
   (`https://fred.stlouisfed.org/series/NIKKEI225`, "Download → CSV"); missing days marked "." are dropped.
2. Set `datafile` at the top of `fit_merton.m` or `fit_kou.m` to the file name.
3. Run the script. It prints the starting values, the maximum-likelihood estimates and the re-estimates on the
   simulated path, followed by the KS test, and plots the two densities.

Parameters are annualised (Δt = 1/252). Example call of the density and simulator:

```matlab
dt = 1/252;  t = linspace(0, 1, 253);
r  = logkournd(dt, 0.08, 0.18, 20, 0.4, 60, 50, t, 1);   % one year of daily returns
f  = logkoupdf(r, dt, 0.08, 0.18, 20, 0.4, 60, 50);       % their densities
```

## Notes

- With a few thousand daily observations the drift μ and the jump intensity λ are estimated imprecisely; the
  volatility and jump-size parameters are better identified. The recovery check in step 3 shows this directly.
- The 2% jump threshold only sets starting values. The maximum-likelihood estimates should not depend on it
  as long as the optimiser reaches the same optimum.
- `fminsearch` is a local optimiser. Trying several starting values is a cheap safeguard.

### Changes in October 2026

The Kou functions were corrected. The density used `normalize` where the standard normal CDF (`normcdf`) is needed,
scaled the diffusion term twice, and had the wrong sign in the down-jump term. The simulator drew normal rather
than double-exponential jumps. The down-jump rate is now estimated as a positive number, *q* is set to 1 − *p*,
invalid parameter values are penalised, and the no-jump density in both models now uses the daily variance σ²Δt.
The corrected densities were checked numerically: they integrate to one, and match histograms of simulated returns.
Scripts were renamed from `workingscript*.m` to `fit_merton.m` and `fit_kou.m`.

## References

- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial
  Economics*, 3(1–2), 125–144.
- Kou, S. G. (2002). A jump-diffusion model for option pricing. *Management Science*, 48(8), 1086–1101.

## Author

**Samuel Edet** · [samueledet.github.io](https://samueledet.github.io)
