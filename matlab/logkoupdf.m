function pdflog = logkoupdf(r,dt,mu,sigma,lambda,p,eta1,eta2)
%LOGKOUPDF Density of one-period log returns under the Kou (2002) model.
%   pdflog = LOGKOUPDF(r,dt,mu,sigma,lambda,p,eta1,eta2) uses the
%   small-dt approximation (at most one jump per period). Up jumps occur
%   with probability p and are exponential with rate eta1; down jumps
%   occur with probability 1-p and are exponential with rate eta2.
%   mu and sigma are the annual drift and volatility, lambda is the
%   annual jump intensity and dt is the period length in years.
    q = 1 - p;
    x = r - (mu - sigma^2/2)*dt;      % return net of diffusion drift
    sd = sigma*sqrt(dt);
    diffusion = normpdf(x, 0, sd);
    if lambda > 0
        up = p*eta1*exp(0.5*sigma^2*eta1^2*dt)*exp(-x*eta1).*normcdf((x - eta1*sigma^2*dt)/sd);
        down = q*eta2*exp(0.5*sigma^2*eta2^2*dt)*exp(x*eta2).*normcdf(-(x + eta2*sigma^2*dt)/sd);
        pdflog = (1 - lambda*dt)*diffusion + lambda*dt*(up + down);
    else
        pdflog = diffusion;
    end
end
