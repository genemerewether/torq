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

function [ M ] = genM( isDFixed )
% Generate the mapping matrix between derivatives for start & end of each
% segment (duplicated) and fixed/optimized derivatives (not duplicated)

%% Arguments:
% isDFixed:

%% Return values:
% M:

%%

numDFixed = nnz(isDFixed);

% fixed derivatives mapping to [start end] list of segment derivatives
i = find(dup(isDFixed));
tmp = zeros(size(isDFixed));
tmp(isDFixed == 1) = 1:numDFixed;
tmp = dup(tmp);
j = tmp(tmp > 0);

% optimized derivatives mapping to [start end] list of segment derivatives
i = [i; find(dup(~isDFixed))];
tmp = zeros(size(isDFixed));
tmp(isDFixed ~= 1) = 1:(numel(isDFixed) - numDFixed);
tmp = dup(tmp);
j = [j; numDFixed + tmp(tmp > 0);];

% create M_trans
% row index is index into dupIndMap
% col index is index into indF, then index into indP
% put a 1 everywhere the value of dupIndMap matches value of indF or indP

v = ones(length(i), 1);
% the row & col indices we generated are for M_trans, so just flip them
M = sparse(j, i, v);

    function A = dup(B)
        if (size(B, 2) == 2)
            A = B;
            return
        end

        A = [B(:,1) repelem(B(:, 2:end-1), 1, 2) B(:,end)];
    end
end
