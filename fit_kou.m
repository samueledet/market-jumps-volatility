%FIT_KOU Fit the Kou (2002) double-exponential jump-diffusion model to daily log returns.
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
jumpup = R > epsilon;
jumpdown = R < -epsilon;
lambdahat = (nnz(jumpup) + nnz(jumpdown))/((length(S)-1)*dt);
Rdiffusion = R(abs(R) <= epsilon);
sigmahat = std(Rdiffusion)/sqrt(dt);
muhat = (2*mean(Rdiffusion) + sigmahat^2*dt)/(2*dt);
phat = nnz(jumpup)/(nnz(jumpup) + nnz(jumpdown));   % probability a jump is up
eta1hat = 1/mean(R(jumpup));                         % rate of up jumps
eta2hat = -1/mean(R(jumpdown));                      % rate of down jumps (positive)

% Maximum likelihood (invalid parameters get an infinite penalty)
theta0 = [muhat sigmahat lambdahat phat eta1hat eta2hat];
options = optimset('MaxFunEvals', 10000, 'MaxIter', 10000);
[theta, fval] = fminsearch(@(th) kounll(R, dt, th), theta0, options);

% Parameter recovery on one simulated path
Rsim = logkournd(dt, theta(1), theta(2), theta(3), theta(4), theta(5), theta(6), t, 1);
thetas = fminsearch(@(th) kounll(Rsim, dt, th), theta, options);

names = {'mu', 'sigma', 'lambda', 'p', 'eta1', 'eta2'};
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
legend('Empirical', 'Kou', 'Location', 'northwest')
title('Density of empirical and simulated log returns')
hold off

function v = kounll(X, dt, th)
%KOUNLL Negative log-likelihood; Inf outside the valid parameter region.
%   eta1 > 1 keeps the expected jump factor E[exp(Y)] finite.
    if th(2) <= 0 || th(3) < 0 || th(4) <= 0 || th(4) >= 1 || th(5) <= 1 || th(6) <= 0
        v = Inf; return
    end
    d = logkoupdf(X, dt, th(1), th(2), th(3), th(4), th(5), th(6));
    if any(d <= 0), v = Inf; else, v = -sum(log(d)); end
end
