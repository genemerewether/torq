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

x = [];
y = [];
z = [];
yaw = [];
xVel = [];
yVel = [];
zVel = [];
xAcc = [];
yAcc = [];
zAcc = [];
%% plot trajectory
for i = 1:nWayPts - 1
    t = 0:0.1:tSeg(i);
    x = [x polyval(P{1}(:, i), t)]; %#ok<AGROW>
    y = [y polyval(P{2}(:, i), t)]; %#ok<AGROW>
    z = [z polyval(P{3}(:, i), t)]; %#ok<AGROW>
    yaw = [yaw polyval(P{4}(:, i), t)]; %#ok<AGROW>

    xVel = [xVel polyval(Pder{i, 1}(:, 2), t)]; %#ok<AGROW>
    yVel = [yVel polyval(Pder{i, 2}(:, 2), t)]; %#ok<AGROW>
    zVel = [zVel polyval(Pder{i, 3}(:, 2), t)]; %#ok<AGROW>

    xAcc = [xAcc polyval(Pder{i, 1}(:, 3), t)]; %#ok<AGROW>
    yAcc = [yAcc polyval(Pder{i, 2}(:, 3), t)]; %#ok<AGROW>
    zAcc = [zAcc polyval(Pder{i, 3}(:, 3), t)]; %#ok<AGROW>
end

xPts = wayPts{1}(1, 1:nWayPts);
yPts = wayPts{2}(1, 1:nWayPts);
zPts = wayPts{3}(1, 1:nWayPts);

figure();
hold on;
plot3(x, y, z); % plot trajectory
xlabel('x (meters)')
ylabel('y (meters)')
zlabel('z (meters)')
view([90,90])

scatter3(xPts, yPts, zPts, 100);

T = zeros(3, size(x, 2));
T(1,:) = xVel;
T(2,:) = yVel;
T(3,:) = zVel;
T = T ./ repmat(sqrt(sum(T.^2, 1)), 3, 1);

rddot = zeros(3, size(x, 2));
rddot(1,:) = xAcc;
rddot(2,:) = xAcc;
rddot(3,:) = xAcc;
% r' x r'' / norm(r' x r'')
B = cross(T, rddot);
B = B ./ repmat(sqrt(sum(B.^2, 1)), 3, 1);

curv = (xVel.*yAcc - yVel.*xAcc)./(xVel.^2 + yVel.^2).^(3/2);
curvRange = 3:length(curv)-2;
curvZeros = zeros(1, length(curvRange));
curv(isnan(curv)) = 0;

% yaw
%quiver3(x, y, z, cos(yaw), sin(yaw), zeros(size(yaw)));

% velocity
quiver3(x, y, z, xVel, yVel, zVel);

% binormal vector
%quiver3(x, y, z, B(1,:), B(2,:), B(3,:));

% acceleration
quiver3(x, y, z, xAcc, yAcc, zAcc);

% display the trajectory
temp = cellfun(@(x) x', P, 'UniformOutput', false);

% curvature using x & y only "flattened curve"
%quiver3(x(curvRange), y(curvRange), z(curvRange), ...
%    curvZeros, curvZeros, curv(curvRange));
