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

%clear

%% settings

tol = 1e-6; % tolerance when checking that polynomials match wayPts

faceFront = 1;
numRepeat = 1;
repeatRatio = 0.9;
% TODO - mereweth@jpl.nasa.gov - change var name to numLoops
assert(numRepeat >= 1, 'numRepeat is number of loops through')

kt = 200; % penalty on total trajectory time

curv2YawVel = 0.01; % scaling factor between curvature at waypoints and desired
% yaw velocities
curv2YawAcc = 1; % scaling factor between curvature at waypoints and desired
% yaw velocities

velDes = 2.5; % desired velocity; used to seed the segment times

% polynomial order 9 necessary for 0th-4th derivative continuity
% polynomial order 5 necessary for 0th-2nd derivative continuity
polyOrder = [9; 9; 5; 5];

nDim = length(polyOrder); % number of polynomials per segment (number of spatial dimensions)
nDer = (polyOrder+1)/2; % number of derivatives for each polynomial

%% record trajectory (xyz position and quaternion rotation)

%record_pose
poses(:, 1) = [ 0  1 -1  1 -1];
poses(:, 2) = [ 0  1  1 -1 -1];
poses(:, 3) = [ 0  0  0  0  0] + 1;
poses(:, 4) = [ 0  0  0  0  0];

% TODO - mereweth@jpl.nasa.gov - check how close end and beginning are
% before joining?

joined_poses = poses;
% join end and beginning
if numRepeat > 1
    joined_poses(end+1,:) = poses(1,:);
end

nWayPtsOne = size(joined_poses, 1);
nWayPts = nWayPtsOne * numRepeat;

if numRepeat > 1
  nWayPts = nWayPts + 1;
end

for i = 1:nDim
    assert(mod(polyOrder(i), 2) == 1, 'polyOrder must be odd')
    cost{i} = zeros(1, nDer(i)); %#ok<SAGROW>
    wayPts{i} = zeros(nDer(i), nWayPts); %#ok<SAGROW>
    isDFixed{i} = zeros(nDer(i), nWayPts); %#ok<SAGROW>
end

% choose cost penalties to minimize control inputs
% x & y moments (corresponding to x & y position) are functions of 4th derivatives of flat outputs 2 & 3
cost{1}(5) = 1;
cost{2}(5) = 1;
cost{3}(3) = 1;
% thrust (corresponding to z position) and z moment (corresponding to yaw) are functions of 2nd derivatives of flat outputs 1 & 4
cost{4}(3) = 1;

%% TODO mereweth@jpl.nasa.gov - set number of waypoints from desired
% discretization?

%% TODO mereweth@jpl.nasa.gov - implement corridor constraints/tubes?

%% TODO mereweth@jpl.nasa.gov - decimate/interpolate to get waypoints

%% Set dummy waypoints for now
%wayPts = cell(1, nDim);
%actWayPtsDup = cell(1, nDim);
%isDFixed = cell(1, nDim);

for i = 1:nDim
    if cost{i}(1)
        warning('Warning: cost penalty on 0th derivative - this is probably not what you want');
    end
end

% TODO - mereweth@jpl.nasa.gov - calculate from list of selected poses
wayPts{1}(1, 1:nWayPtsOne) = joined_poses(1:end, 1); % x
wayPts{2}(1, 1:nWayPtsOne) = joined_poses(1:end, 2); % y
wayPts{3}(1, 1:nWayPtsOne) = joined_poses(1:end, 3); % z

% if faceFront is true, this will be overwritten
wayPts{4}(1, 1:nWayPtsOne) = joined_poses(1:end, 4); % yaw

for i = 1:nDim
    % zero-th derivatives are xyz position & yaw values
    % by default, only zero-th derivatives are set at waypoints
    isDFixed{i}(1, 1:nWayPtsOne) = ones(1, nWayPtsOne);

    % and by default, the start and end points for the entire trajectory have
    % zeros for every derivative 1st order and higher
    isDFixed{i}(1:end, 1) = ones(nDer(i), 1);
    isDFixed{i}(1:end, nWayPtsOne) = ones(nDer(i), 1);
end

% repeat
% for i = 1:nDim
%     wayPts{i}(1, nWayPtsOne+1:nWayPts) = ...
%         [repmat(wayPts{i}(1, 1:nWayPtsOne), 1, numRepeat-1), wayPts{i}(1,1)];
%
%     isDFixed{i}(:, nWayPtsOne+1:nWayPts-1) = ...
%         repmat(isDFixed{i}(:, 1:nWayPtsOne), 1, numRepeat-1);
%
%     isDFixed{i}(1:end, end) = ones(nDer(i), 1);
% end


%% Setup to formulate QP for [x,y,z,yaw]

% for yaw, also control first derivatives at each waypoint
if faceFront
    isDFixed{nDim}(2, :) = ones(1, nWayPts);
end

% handle yaw separately
% isDFixed{4}(1, :) = zeros(1, nWayPts);
% isDFixed{4}(:, 1) = ones(nDer(4), 1);
% isDFixed{4}(:, end) = ones(nDer(4), 1);
% isDFixed{4}(1, 1) = 1;
% isDFixed{4}(1, end) = 1;

for i = 1:nDim
    if any(wayPts{i}(~isDFixed{i}))
        warning('Warning: some floating derivatives were specified in wayPts - these values are ignored');
    end
end

% initial segment times based on desired average velocity
wayPtDeltas = diff([wayPts{1}(1, 1:nWayPts); wayPts{2}(1, 1:nWayPts); wayPts{3}(1, 1:nWayPts)], 1, 2);

tSeg = sqrt(sum(wayPtDeltas.^2, 1)) / velDes;
tSeg(end) = 1;
%tSeg = ones(1, nWayPts-1);

if numRepeat > 1
    for i = 0:numRepeat-1
        tSeg(i*nWayPtsOne+1:(i+1)*nWayPtsOne) = repeatRatio^i * tSeg(i*nWayPtsOne+1:(i+1)*nWayPtsOne);
    end
end

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

%% optimize relative segment timing; loop until threshold on gradient is reached

[tSeg,costFinal,exitFlag] = fminunc(@(X) genPoly(wayPts, X, cost, polyOrder, isDFixed, kt, faceFront), tSeg);

[ ~, P ] = genPoly(wayPts, tSeg, cost, polyOrder, isDFixed, kt);

% calculate derivatives up to 2nd order
Pder = cell(nWayPts-1, nDim);
for i = 1:nDim
    for j = 1:nWayPts-1
        [~, Pder{j, i}] = polyderN(P{i}(:, j), nDer(i));
    end
end

if faceFront
    % align yaw with trajectory
    myeps = 0.2;
    % Pder(1:end-1,1) is because of repeated terminal waypoint for each
    % repeat - have to use the 3rd-to-last velocity for the 2nd-to-last and last

    % Pder(1:end-1,1) is x, Pder(1:end-1,2) is y

    % polyval(x(:,2),myeps) means 1st derivative
    wayPtVel = [cellfun(@(x) polyval(x(:,2),myeps), Pder(1:end-1,1)), ... % x vel
        cellfun(@(x) polyval(x(:,2),myeps), Pder(1:end-1,2))]; % y vel

    if length(tSeg) > 1
        %wayPtVel(end+1, :) = [polyval(Pder{end, 1}(:,2),tSeg(end-1)-myeps), ...
        %    polyval(Pder{end, 2}(:,2),tSeg(end-1)-myeps)]; %#ok<SAGROW>
        wayPtVel(end+1:end+2, :) = repmat([polyval(Pder{end-1, 1}(:,2),tSeg(end-1)-myeps), ...
            polyval(Pder{end-1, 2}(:,2),tSeg(end-1)-myeps)], 2, 1); %#ok<SAGROW>
    else
        wayPtVel(end+1, :) = [polyval(Pder{1, 1}(:,2),tSeg(1)-myeps), ...
            polyval(Pder{1, 2}(:,2),tSeg(1)-myeps)]; %#ok<SAGROW>
    end

    % order is [x, y] in wayPtVel
    wayPts{nDim}(1, 1:nWayPts) = atan2(wayPtVel(:, 2), wayPtVel(:, 1));

    % use curvature to set higher order derivatives of yaw
    % polyval(x(:,3),myeps) means 2nd derivative
    wayPtAcc = [cellfun(@(x) polyval(x(:,3),myeps), Pder(1:end-1,1)), ... % x acc
        cellfun(@(x) polyval(x(:,3),myeps), Pder(1:end-1,2))]; % y acc

    if length(tSeg) > 1
        %wayPtAcc(end+1, :) = [polyval(Pder{end, 1}(:,3),tSeg(end)-myeps), ...
        %    polyval(Pder{end, 2}(:,3),tSeg(end)-myeps)]; %#ok<SAGROW>
        wayPtAcc(end+1:end+2, :) = repmat([polyval(Pder{end-1, 1}(:,3),tSeg(end-1)-myeps), ...
            polyval(Pder{end-1, 2}(:,3),tSeg(end-1)-myeps)], 2, 1); %#ok<SAGROW>
    else
        wayPtAcc(end+1, :) = [polyval(Pder{1, 1}(:,3),tSeg(1)-myeps), ...
            polyval(Pder{1, 2}(:,3),tSeg(1)-myeps)]; %#ok<SAGROW>
    end

    % order is [x, y] in wayPtAcc
    curv = (wayPtVel(:,1).*wayPtAcc(:,2) - wayPtVel(:,2).*wayPtAcc(:,1)) ...
        ./ (wayPtVel(:,1).^2 + wayPtVel(:,2).^2) .^ (3/2);

    wayPts{4}(2,:) = curv2YawVel * curv;

    % TODO - mereweth@jpl.nasa.gov - check/fix this 2 pi wrapping function
    % for more use cases
    wayPts{4}(1,:) = (wayPts{4}(1,:)+pi<pi)*2*pi + wayPts{4}(1,:);

    [ ~, P(nDim) ] = genPoly(wayPts(nDim), tSeg, cost(nDim), polyOrder(nDim), isDFixed(nDim), kt);
end

%% TODO - mereweth@jpl.nasa.gov - simulate trajectory and check for actuator saturation

plotTraj
%send_poly_traj

%% plot and generate header
% try
%     names = cell(1, length(polyOrder));
%     names{1} = 'x'; names{2} = 'y'; names{3} = 'z'; names{4} = 'yaw';
%     fd=fopen('~/snappy/eagle_dev/Firmware/src/modules/polyeval/polyeval_test_traj.h','wt');
%     makeHeader(fd, P, tSeg, names);
%     fclose(fd);
%     fd=fopen('~/snappy/Firmware/src/modules/polyeval/polyeval_test_traj.h','wt');
%     makeHeader(fd, P, tSeg, names);
%     fclose(fd);
% catch
%     disp 'Invalid file name for generated header output'
% end
