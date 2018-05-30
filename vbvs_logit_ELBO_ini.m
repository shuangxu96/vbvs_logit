% C. Zhang, S. Xu and J. Zhang. A Novel Variational Bayesian Method for
% Variable Selection in Logistic Regression Models. 2018

function ELBO = vbvs_logit_ELBO_ini(mu, invSigma, theta, rho, a, b0, b, N)
% Compute evidence lower bound in the initialization stage
theta(theta==0) = eps;
theta(theta==1) = 1-eps;
ELBO = 0.5 * (mu'*invSigma*mu - logdet(invSigma)) + ...
sum(theta.*log(rho./theta) + (1-theta).*log((1-rho)./(1-theta))) - ...
sum(a.*(b0./b + log(b))) - N * log(2);
