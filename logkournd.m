function Rsim = logkournd(dt,mu,sigma,lambda,p,eta1,eta2,t,Ns)
%LOGKOURND Simulate log returns under the Kou (2002) model.
%   Rsim = LOGKOURND(dt,mu,sigma,lambda,p,eta1,eta2,t,Ns) returns a
%   (length(t)-1)-by-Ns matrix of log returns. Jump sizes are
%   double-exponential: +Exp(eta1) with probability p, -Exp(eta2) otherwise.
n = length(t) - 1;
dN = poissrnd(lambda*dt, n, Ns);                   % number of jumps
dW = sqrt(dt).*randn(n, Ns);                       % Brownian increments
Y = zeros(n, Ns);                                  % total jump per period
for k = 1:max(dN(:))
    idx = dN >= k;
    m = nnz(idx);
    isup = rand(m,1) < p;
    jump = exprnd(1/eta1, m, 1).*isup - exprnd(1/eta2, m, 1).*~isup;
    Y(idx) = Y(idx) + jump;
end
Rsim = (mu - sigma^2/2)*dt + sigma*dW + Y;
end
