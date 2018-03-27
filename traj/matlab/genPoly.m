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

function [ totalCost, P ] = genPoly(wayPts, tSeg, cost, polyOrder, isDFixed, kt, faceFront)
% Generate polynomial trajectory
% returns coefficients and cost

%% Arguments:
% tSeg:
% cost:
% polyOrder:
% isDFixed:
% kt:

%% Return values:
% P:

%% helper variables
totalCost = 0;

nDim = length(polyOrder); % number of polynomials per segment (number of spatial dimensions)
nDer = (polyOrder+1)/2; % number of derivatives for each polynomial
numDFixed = zeros(nDim, 1);
DF = cell(1, nDim);
M = cell(1, nDim);

for i = 1:nDim
    M{i} = genM(isDFixed{i});

    indDFixed = find(isDFixed{i});
    numDFixed(i) = length(indDFixed);

    DF{i} = wayPts{i}(indDFixed);
end

Q = cell(1, nDim);
A = cell(1, nDim);
R = cell(1, nDim);
DP = cell(1, nDim);

%% Solve for free derivatives at segment boundaries
for i = 1:nDim
    Q{i}= genQ(tSeg, cost{i}, polyOrder(i));
    A{i} = genA(tSeg, polyOrder(i));
    R{i} = M{i} / A{i}' * Q{i} / A{i} * M{i}';
    %R{i} = M{i} * pinv(full(A{i}))' * Q{i} * pinv(full(A{i})) * M{i}';

    % top right section
    RFP = R{i}(1:numDFixed(i), (numDFixed(i)+1):end);
    RPP = R{i}((numDFixed(i)+1):end, (numDFixed(i)+1):end);

    % equation 31 in the paper
    DP{i} = -RPP \ RFP' * DF{i};
    %DP{i} = -inv(RPP) * RFP' * DF{i};
end

%% Recover polynomial coefficients
P = cell(1, nDim);
for i = 1:nDim
    nCoeff = polyOrder(i)+1;
    p = A{i} \ M{i}' * [DF{i}; DP{i}];

    totalCost = totalCost + p' * Q{i} * p;

    P{i} = flip(reshape(flip(p), nCoeff, []), 2);
end

totalCost = totalCost + kt * sum(tSeg);

end
