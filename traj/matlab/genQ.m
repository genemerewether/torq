% Copyright 2016, by the California Institute of Technology. ALL RIGHTS
% RESERVED. United States Government Sponsorship acknowledged. Any commercial
% use must be negotiated with the Office of Technology Transfer at the
% California Institute of Technology.
%
% This software may be subject to U.S. export control laws. By accepting this
% software, the user agrees to comply with all applicable U.S. export laws and
% regulations. User has the responsibility to obtain export licenses, or other
% export authority as may be required before exporting such information to
% foreign countries or providing access to foreign persons.

function [ Q ] = genQ( tSeg, cost, polyOrder )
% Generate the weighted sum of Hessian matrices for each of the polynomial
% derivatives for all segments.

%% Arguments:
% tSeg:
% cost:
% polyOrder:

%% Return values:
% Q:

%%

nCoeff = polyOrder + 1;
nSeg = length(tSeg);

% derivatives from r = 0 to polyOrder where cost is not zero
der = find(cost) - 1;
tPower = zeros(nCoeff, nCoeff, length(der));
prd = zeros(nCoeff, nCoeff, length(der));

% this for loop is OK - der has < nCoeff/2 entries
for derInd = 1:length(der)
    r = der(derInd); % what order derivative
    if r ~= 0
        m = repmat((0:r-1)', 1, nCoeff); % m in the paper
        i = repmat([zeros(1, r), r:nCoeff-1], r, 1); % i and l in the paper
        iMinusM = i-m; % i-m
        % iMinusM(iMinusM < 0) = 0; % don't need? no negative coefficients
        tempPrd = prod(iMinusM, 1); % premultiply all i-m
        prd(:,:,derInd) = tempPrd'*tempPrd; % (i-m) * (l-m)
    else
        % empty product because r=0-1 is less than m=0
        prd(:,:,derInd) = ones(nCoeff);
    end

    [row, col] = meshgrid(0:nCoeff-1, 0:nCoeff-1);
    tPower(:,:,derInd) = row+col+ones(size(row))*(-2*r+1); % raise tSeg to this power
end

tPowerInd = find(tPower > 0);

blockDerQ = zeros([nCoeff, nCoeff, length(der), nSeg]);

% set up indices for each time segment (1st repmat), adding offset (2nd repmat)
blockDerQInd = repmat(tPowerInd, 1, nSeg)+repmat((0:nSeg-1)*nCoeff^2*length(der), length(tPowerInd), 1);

% only calculate indices for which tPower > 0 to prevent divide by 0
% equation 14 in the paper
blockDerQ(blockDerQInd) = 2 * repmat(tSeg, length(tPowerInd), 1) .^ repmat(tPower(tPowerInd), 1, nSeg) ...
    .* repmat(prd(tPowerInd) ./ tPower(tPowerInd), 1, nSeg);

% weighted sum over costs (third dimension) here
% equation 15 in the paper
blockQ = sum(blockDerQ .* repmat(reshape(cost(der+1), 1, 1, []), nCoeff, nCoeff, 1, nSeg), 3);

% each segment has same block in block-diagonal matrix except for tSeg
blockQ = permute(blockQ, [1 2 4 3]);
nzInd = find(blockQ > 0);
[i, j, k] = ind2sub(size(blockQ), nzInd);
i = (k-1)*nCoeff+i;
j = (k-1)*nCoeff+j;

% fast way but not sure it works every time:
% only copy square from min deriv to bottom right corner
%i= reshape(repmat(row, 1, nSeg) + repmat((0:nSeg-1)*nCoeff, length(row), 1), [], 1);
%j = reshape(repmat(col, 1, nSeg) + repmat((0:nSeg-1)*nCoeff, length(row), 1), [], 1);

v = blockQ(nzInd);

Q = sparse(i, j, v, nCoeff*nSeg, nCoeff*nSeg);

end
