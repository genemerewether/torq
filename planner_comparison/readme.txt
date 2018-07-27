This folder contains data relevant to the tests presented in the paper: "Comparison of Trajectory Optimization Algorithms for High-Speed Quadrotor Flight Near Obstacles".
These tests include batch simulations and flight tests. 

Included in these folders are the waypoint sets, obstacle representations (ESDF) and 3D mesh representations of the environments. 
The waypoints for the batch tests have two sets - one that is the output from RRT (a small set of waypoints), and one with a dense set of waypoints (needed for UNCO and hence labelled as such).
E.g. the sparse one is: rand_waypoints_198-B4_V3_001.yaml
And the dense one is: rand_waypoints_198-B4_V3_001_unco.yaml
For the flight tests there is only one set of waypoints. These are a dense set and should be simplified using an epsilon of 0.6 or 0.7 in the RDP widget when using the waypoint set. 

These data are intended to be used with the TORQ GCS, and associated trajectory optimisation algorithm implementations: https://github.com/genemerewether/torq/tree/dev/planner_comparison_2018_07

-------------
IMPORTANT:
-------------
Use the "planner_comparison_2018_07" branch for the tests. 

FOLDERS
Seg_times - optimal segment times used 
Results - results from tests run that were presented in the paper


----------------------
----------------------
SCRIPTS
----------------------
----------------------
There are two files used for running the batch tests on the waypoints

1) run_planners.py
	- This runs the batch planning instances on the environments (one environment at a time)
2) test_data.py
	- This compiles together the results from the tests to get overall metrics (can also be used to look into data in detail)


-------------------------------
Notes for run_planners.py
-------------------------------
Lines to change for the filepath
1) 584-595 - for loading the ESDF and waypoints
2) 598 to load the paramters for computing performance metrics (not essential if remove performance metrics)
3) Lines 618, 621 - changing the filepath to the optimal segment times
4) 709 - 719 - Filepaths w runshen opening files for saving

To change to control the running
1) Change the number of tests to run 
	Line 640
2) To run the time optimisation 
	Line 601 - Change boolean
	Will only run time optimisation
3) Save trajectories
	- Uncomment the block from 689 down. 
	

-------------------------------
Notes for test_data.py
-------------------------------
Processes data files that get unpickled and are stored in .data files (custom file extension), e.g. unco_198-B4_V3_03_data.data
Examples are included in the directory

These are expected to be in the "Results" folder in the current working directory

There are a few places to change filenames/filepaths and change which environment is being analysed
1) Change environment
	a) Uncomment one of line 279/280
	b) Uncomment one of line 324/325
2) Change output filename in lines 279/280
3) Comment out line 335 to not check continuity


