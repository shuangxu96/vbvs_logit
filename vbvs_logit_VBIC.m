% C. Zhang, S. Xu and J. Zhang. A Novel Variational Bayesian Method for
% Variable Selection in Logistic Regression Models. 2018

function VBIC = vbvs_logit_VBIC(theta, p, logdetinvSigma, a, b)
% Compute variational BIC.

theta(theta==0) = eps;
theta(theta==1) = 1-eps;
VBIC = -0.5 * (p*(log(2*pi)+1) - logdetinvSigma) + ...
    sum( (a-1).*psi(a) + log(b) - a - gammaln(a) + theta.*log(theta) + (1-theta).*log(1-theta) );