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

function [ A ] = genA( tSeg, polyOrder )
% Generate the constraint matrix for beginning and end of each segment.
% Maps between polynomial coefficients and derivatives evaluated at
% beginning and end of segments.

%% Arguments:
% tSeg:
% polyOrder:

%% Return values:
% A:

%%
nCoeff = polyOrder + 1;
nDer = floor(nCoeff/2);
nSeg = length(tSeg);

[n, r] = meshgrid(0:nCoeff-1, 0:nDer-1);
nMinusR = n - r;

% nMinusR < 0 means r < n, this value is automatically 0
nMinusRInd = find(nMinusR >= 0);
if size(nMinusRInd, 1) == 1
    nMinusRInd = nMinusRInd';
end
tPower = zeros(nDer, nCoeff, nSeg);
% set up indices for each time segment (1st repmat), adding offset (2nd repmat)
tPowerInd = repmat(nMinusRInd, 1, nSeg)+repmat((0:nSeg-1)*nDer*nCoeff, size(nMinusRInd, 1), 1);
% raise tSeg to nMinusR power
temp = nMinusR(nMinusRInd);
if size(temp, 1) == 1
    temp = temp';
end
tPower(tPowerInd) = repmat(tSeg, length(nMinusRInd), 1) .^ repmat(temp, 1, nSeg);

% construct the product of (r-m) from m = 0 to r-1 as the rows of rMinusM
rMinusM = repmat([ones(1, nCoeff); nMinusR(1:end-1, :)], 1, 1, nDer); % can reuse meshgrid from above
rMinusM(rMinusM < 1) = 1;
[j,i,k] = meshgrid(1:nCoeff, 1:nDer, 1:nDer);
oneInd = find(i>k); % these should be 1
rMinusM(sub2ind(size(rMinusM), i(oneInd), j(oneInd), k(oneInd))) = 1;
prd = permute(prod(rMinusM, 1), [3,2,1]); % multiply down each column
if nDer == 1
    if nSeg <= 2
        A0 = prd(1:nSeg)';
    else
        A0 = prd(1:end)';
    end
else
    A0 = repmat(diag(prd), nSeg, 1); % repeat diagonal entries
end
AT = repmat(prd, 1, 1, nSeg) .* tPower;

% only copy diag of A0 and upper triangle of AT for each segment
A0Row = (1:nDer*nSeg) + (repelem((0:nSeg-1)*nDer, nDer));
if mod(polyOrder, 2) == 0
    A0Col = A0Row + repelem(0:nSeg-1, nDer);
else
    A0Col = A0Row;
end

% get upper triangular row & column indices
[i, j] = ind2sub(size(nMinusR), nMinusRInd');
ATVal = AT(sub2ind(size(AT), repmat(i, 1, nSeg), repmat(j, 1, nSeg), repelem(1:nSeg, length(i))));
ATRow = repmat(i, 1, nSeg) + (repelem((0:nSeg-1)*2*nDer, length(i)))+nDer;
ATCol = repmat(j, 1, nSeg) + (repelem((0:nSeg-1)*nCoeff, length(j)));

A = sparse([A0Row ATRow], [A0Col ATCol], [A0' ATVal]);

end
