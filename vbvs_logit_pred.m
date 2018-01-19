function out = vbvs_logit_pred(X, model)
%% out = vbvs_logit_pred(X, model)
%
% Return the predictions on X by model.
%
% Copyright (c) 2017, Shuang Xu
% All rights reserved.
% See the file LICENSE for licensing information.

%% 
n = length(X(:,1));
if (model.intercept)
    X = [ones(n,1),X];
end
Importance = model.Importance; ind = find(Importance~=0);
mu = model.mu(ind);
Sigma = model.Sigma(ind,ind);
invSigma = model.invSigma(ind,ind);
X = X(:,ind);
out = vb_logit_pred(X, mu, Sigma, invSigma);

