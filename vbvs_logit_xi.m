% C. Zhang, S. Xu and J. Zhang. A Novel Variational Bayesian Method for
% Variable Selection in Logistic Regression Models. 2018

function [xi, delta_xi, S] = vbvs_logit_xi(X, D_w)
% Update local variational parameter.

xi = sqrt(sum(X .* (X * D_w), 2));
delta_xi = tanh(xi ./ 2) ./ (4 .* xi);
delta_xi(isnan(delta_xi)) = 1/8;
S = X' * bsxfun(@times, X, delta_xi);