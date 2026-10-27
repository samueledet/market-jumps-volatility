function pdflog = logmertonpdf(r,dt,mu,sigma,lambda,mu_j,sigma_j)
%LOGMERTONPDF Density of one-period log returns under the Merton (1976) model.
%   pdflog = LOGMERTONPDF(r,dt,mu,sigma,lambda,mu_j,sigma_j) is a
%   Poisson mixture of normals: with k jumps in the period, the log return
%   is normal with mean (mu-sigma^2/2)*dt + k*mu_j and variance
%   sigma^2*dt + k*sigma_j^2. The sum is truncated at nterm jumps.
    m = (mu - sigma^2/2)*dt;
    if lambda > 0
        nterm = 100;
        pdflog = zeros(size(r));
        for k = 0:nterm
            poisson = (lambda*dt)^k/factorial(k)*exp(-lambda*dt);
            pdflog = pdflog + poisson*normpdf(r, m + mu_j*k, sqrt(sigma^2*dt + sigma_j^2*k));
        end
    else
        pdflog = normpdf(r, m, sigma*sqrt(dt));
    end
end
