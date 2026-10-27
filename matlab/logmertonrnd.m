function Rsim = logmertonrnd(dt,mu,sigma,lambda,mu_j,sigma_j,t,Ns)
%LOGMERTONRND Simulate log returns under the Merton (1976) model.
%   Rsim = LOGMERTONRND(dt,mu,sigma,lambda,mu_j,sigma_j,t,Ns) returns a
%   (length(t)-1)-by-Ns matrix of log returns. Jump sizes are normal with
%   mean mu_j and standard deviation sigma_j.
dN = poissrnd(lambda*dt, length(t)-1, Ns);                 % number of jumps
Y = mu_j*dN + sigma_j*sqrt(dN).*randn(length(t)-1, Ns);    % total jump
dW = sqrt(dt).*randn(length(t)-1, Ns);                     % Brownian increments
Rsim = (mu - sigma^2/2)*dt + sigma*dW + Y;
end
