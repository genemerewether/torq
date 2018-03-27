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

function [ q, allDer ] = polyderN( p, n )
% Calculates derivatives of a polynomial from 0th up to nth, and returns
% them.

%% Arguments:
% p:
% n:

%% Return values:
% q:
% allDer:

%%

allDer = zeros(length(p), n+1);

allDer(:, 1) = p;
for i = 1:n
    p = polyder(p)';
    allDer(:, i+1) = [zeros(size(allDer, 1) - length(p), 1); p];
end

q = p;

end
