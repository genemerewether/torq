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

function [ ] = makeHeader( fd, P, tSeg, names )
% Write all variables to file fd in C header format for quick tests

%% Arguments:
% fd:
% P:
% names:
% tSeg:

%% Return values:

%%
tSeg = cumsum(tSeg);

fprintf(fd, '#include "polytraj_var_size.h"\n\n');
%fprintf(fd, '#define POLYTRAJ_FLOAT double\n\n');

for i = 1:length(P)
    fprintf(fd, 'double _%s_t_transition[%d] = ', names{i}, length(tSeg));
    literalArray(tSeg);
    fprintf(fd, ';\nPOLYTRAJ_FLOAT _%s_coeffs[%d] = ', names{i}, numel(P{i}));
    literalArray(P{i});
    fprintf(fd, ';\n');

    fprintf(fd, 'Traj _%s_traj(%d, _%s_t_transition, %d, _%s_coeffs);\n', ...
        names{i}, length(tSeg), names{i}, size(P{i}, 1), names{i});
end

    function literalArray(A)
        fprintf(fd, '{%.20g', A(1));
        fprintf(fd, ', %.20g', A(2:end));
        fprintf(fd, '}');
    end

end
