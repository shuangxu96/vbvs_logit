% C. Zhang, S. Xu and J. Zhang. A Novel Variational Bayesian Method for
% Variable Selection in Logistic Regression Models. 2018

function [ELBO, logdetinvSigma] = vbvs_logit_ELBO(mu, invSigma, theta, rho, xi, delta_xi, a, b0, b, logdetinvSigma)
% Compute evidence lower bound

theta(theta==0) = eps;
theta(theta==1) = 1-eps;
ELBO = 0.5 * (mu'*invSigma*mu - logdetinvSigma - sum(xi)) + ...
sum(theta.*log(rho./theta) + (1-theta).*log((1-rho)./(1-theta))) - ...
sum(a.*(b0./b + log(b))) + sum(delta_xi.*(xi.^2) + logsigmoid(xi));
