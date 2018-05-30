% C. Zhang, S. Xu and J. Zhang. A Novel Variational Bayesian Method for
% Variable Selection in Logistic Regression Models. 2018

function model = vbvs_logit_fit(X, y, rho, a0, b0, maxiter, tol)
% Infer the model for a single rho.

%% pre-compute some constants
[N, p] = size(X);
X = [ones(N,1), X]; p = p + 1;
lam = log(rho./(1-rho));
theta = ones(p,1);
a0 = [1e-2;a0]; b0 = [1e-4;b0];
a = a0 + 0.5;
t_mu = X'*y;
delta_xi = ones(N, 1) / 8;
E_a = a0 ./ b0;
S = X' * bsxfun(@times, X, delta_xi);

[theta, ~, invSigma, mu, ~, D_w, b, E_a] = vbvs_logit_update(theta,S,E_a,t_mu,b0,a,lam,p);
ELBO_last = vbvs_logit_ELBO_ini(mu, invSigma, theta, rho, a, b0, b, N);
MU = mu;

ELBO = zeros(maxiter,1);
logdetinvSigma = [];
Sigma = [];
converge = false; t = 1; 
while ~converge
    % update xi 
    [xi, delta_xi, S] = vbvs_logit_xi(X, D_w);
    % update posterior parameters of a based on xi
    [theta, ~, invSigma, mu, Sigma, D_w, b, E_a] = vbvs_logit_update(theta,S,E_a,t_mu,b0,a,lam,p);
    logdetinvSigma = logdet(invSigma);
    ELBO(t) = vbvs_logit_ELBO(mu, invSigma, theta, rho, xi, delta_xi, a, b0, b, logdetinvSigma);
    MU = [MU, mu];
    % converge or not
    if t>= maxiter;converge = true;
    elseif abs(ELBO_last - ELBO(t)) < tol;converge = true;end
    %ELBO_last = ELBO(t);
    t = t + 1;
end
ELBO = ELBO(1:(t-1));
ELBO = ELBO + sum(a0.*(1+log(b0)) + gammaln(a)-gammaln(a0) + 0.5);
VBIC = vbvs_logit_VBIC(theta, p, logdetinvSigma, a, b);
gamma = theta;
gamma(gamma<0.5) = 0; gamma(gamma>=0.5) = 1;
gamma = logical(gamma);
p1 = sum(gamma~=0);
coef = zeros(size(gamma));
coef(gamma) = mu(gamma);
BIC = 2 * sum(log(1+exp( - y .* (X * coef)))) + p1 * log(N);

%% output
model.ELBO = ELBO';
model.VBIC = VBIC;
model.BIC = BIC;
model.mu = mu;
model.coef = coef;
model.Sigma = Sigma;
model.invSigma = invSigma;
model.Step = t;
model.theta = theta;
model.lambda = lam;
model.muHistory = MU;
model.gamma = gamma;
model.converge_flag = (t < maxiter);
