% logpexp(x) returns log(1 + exp(x)). The computation is performed in a
% numerically stable manner. For large entries of x, log(1 + exp(x)) is
% effectively the same as x.
function y = logpexp (x)

  % Part of the varbvs package, https://github.com/pcarbo/varbvs
  %
  % Copyright (C) 2012-2017, Peter Carbonetto
  %
  % This program is free software: you can redistribute it under the
  % terms of the GNU General Public License; either version 3 of the
  % License, or (at your option) any later version.
  %
  % This program is distributed in the hope that it will be useful, but
  % WITHOUT ANY WARRANY; without even the implied warranty of
  % MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  % General Public License for more details.
  %
  y    = x;
  i    = find(x < 16);
  y(i) = log(1 + exp(x(i)));
