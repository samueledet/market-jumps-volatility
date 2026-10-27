%FIT_MERTON Fit the Merton (1976) jump-diffusion model to daily log returns.
%   Steps: (1) threshold estimates as starting values, (2) maximum
%   likelihood, (3) simulate a path from the fitted model and re-estimate
%   on it, to check the parameters can be recovered, (4) compare empirical
%   and simulated returns with a two-sample Kolmogorov-Smirnov test and
%   kernel density plots.

datafile = 'japan_stockprice.csv';   % see README for data
S = load_prices(datafile);
dt = 1/252;                           % daily data, 252 trading days a year
t = linspace(0, (length(S)-1)*dt, length(S));
R = diff(log(S));

% Starting values: returns above epsilon in absolute value are treated as jumps
epsilon = 0.02;
jumpindex = abs(R) > epsilon;
lambdahat = nnz(jumpindex)/((length(S)-1)*dt);
Rjumps = R(jumpindex);
Rdiffusion = R(~jumpindex);
sigmahat = std(Rdiffusion)/sqrt(dt);
muhat = (2*mean(Rdiffusion) + sigmahat^2*dt)/(2*dt);
sigma_jhat = sqrt(max(var(Rjumps) - sigmahat^2*dt, eps));
mu_jhat = mean(Rjumps) - (muhat - sigmahat^2/2)*dt;

% Maximum likelihood (invalid parameters get an infinite penalty)
theta0 = [muhat sigmahat lambdahat mu_jhat sigma_jhat];
options = optimset('MaxFunEvals', 10000, 'MaxIter', 10000);
[theta, fval] = fminsearch(@(th) mertonnll(R, dt, th), theta0, options);

% Parameter recovery on one simulated path
Rsim = logmertonrnd(dt, theta(1), theta(2), theta(3), theta(4), theta(5), t, 1);
thetas = fminsearch(@(th) mertonnll(Rsim, dt, th), theta, options);

names = {'mu', 'sigma', 'lambda', 'mu_j', 'sigma_j'};
fprintf('%-8s %10s %10s %14s\n', '', 'start', 'MLE', 'sim. re-est.');
for i = 1:numel(names)
    fprintf('%-8s %10.4f %10.4f %14.4f\n', names{i}, theta0(i), theta(i), thetas(i));
end

% Do empirical and simulated returns come from the same distribution?
[h, pval] = kstest2(R, Rsim);
fprintf('KS test: h = %d, p-value = %.3f\n', h, pval);

[f, xi] = ksdensity(R);
plot(xi, f, '-r', 'LineWidth', 2); hold on
[g, yi] = ksdensity(Rsim);
plot(yi, g, '-b', 'LineWidth', 2)
legend('Empirical', 'Merton', 'Location', 'northwest')
title('Density of empirical and simulated log returns')
hold off

function v = mertonnll(X, dt, th)
%MERTONNLL Negative log-likelihood; Inf outside the valid parameter region.
    if th(2) <= 0 || th(3) < 0 || th(5) <= 0
        v = Inf; return
    end
    d = logmertonpdf(X, dt, th(1), th(2), th(3), th(4), th(5));
    if any(d <= 0), v = Inf; else, v = -sum(log(d)); end
end
