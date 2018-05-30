% C. Zhang, S. Xu and J. Zhang. A Novel Variational Bayesian Method for
% Variable Selection in Logistic Regression Models. 2018

function [theta, Omega, invSigma, mu, Sigma, D_w, b, E_a] = vbvs_logit_update(theta,S,E_a,t_mu,b0,a,lam,p)
% Update the variational distribution.

Omega = theta*theta'+diag(theta.*(1-theta));
invSigma = diag(E_a) + 2 * S .* Omega;
Sigma = inv(invSigma);
mu = Sigma * bsxfun(@times, 0.5 * t_mu, theta);
D_w = Sigma + mu * mu';
b = b0 + 0.5 * diag(D_w);
E_a = a./b;
tempt1 = mu .* t_mu / 2;
tempt2 = diag(S) .* diag(D_w);
tempt3 = zeros(p,1);
for k=1:p
    tempt3(k,1) = sum(S(k,:)' .* theta .* D_w(:,k));
end
tempt3 = tempt3 - tempt2 .* theta;
theta = 1./(1+exp(-(lam + tempt1 - tempt2 -2 * tempt3)));
theta(1) = 1;