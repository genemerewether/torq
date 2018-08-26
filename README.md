# Summary
This branch of the repository presents the implementation of three trajectory optimization algorithms that were used in the tests presented in the Robotics and Automation Letter: "Comparison of Trajectory Optimization Algorithms for High-Speed Quadrotor Flight Near Obstacles". The paper presents a comparative analysis between three algorithms for optimising quadrotor trajectories for high-speed flight indoors, near to obstacles. 

Below and in the folders are also links to the datasets used for the tests, including waypoint files and obstacle representations. 

The code presented here includes a GUI for planning, observing and manipulating trajectories. 

The abstract of the paper is included below, before a summary of the inclusions to the torq repository for the planner comparison and then installation and running instructions. 

-- Repo notes:

Ground control station and optimization code from Tango on Quadrotors project - NTR 50759

Branch - planner_comparison_2018_07 - to include data and code for running the comparison of trajectory planning algorithms as presented in "Comparison of Trajectory Optimization Algorithms for High-Speed Quadrotor Flight Near Obstacles"

# Abstract of paper
For autonomous quadrotors to be used for applications such as delivery, first responder search, and inspection, there is a need to fly near obstacles, especially in urban and indoor environments. There is also benefit to fly at high speeds, to complete the given task in a short period of time. A key challenge in enabling these capabilities is to quickly plan trajectories that are safe, collision-free, and dynamically feasible. Dynamic feasibility in this context means that the trajectory can be accurately tracked in flight: a critical requirement when flying near to obstacles. 
It is important to understand the relative strengths and weaknesses of trajectory planning algorithms; hence we compare three state-of-the-art algorithms (Bry et al. 2015, Campos-Macias et al. 2017 and Chamitoff et al. 2018) with both a set of planning scenarios and an assessment of performance when the trajectories are tracked in flight. 
We introduce a combination of the work of Campos-Macias et al. with Bry et al., which is shown to perform well for slow speed, conservative trajectories. The algorithm from Bry et al. is implemented and shown to be the best for higher speed trajectories with relatively large amounts of open space. An adaptation of the algorithm from Chamitoff et al. for quadrotor applications, as presented here, is the best for high-speed trajectories in tight confines and relaxes the requirement for a known collision-free path to seed the optimization.


# Additions for planner comparison
The additions are in the planner_comparison folder. 
1. Waypoints for the test cases
2. ESDF folder as the obstacle representations and 3D meshes (files are here: https://www.dropbox.com/sh/rahfytrn0ketqnz/AACa0KhhJI25hsauFhokvw13a?dl=0)
3. Scripts to run the batch comparisons and analyse the output results
4. Example output results as used in the paper 

Readme.txt files in the folders and sub-folders containt more information. 

Below are the general instructions for installing and operating the TORQ GCS, which includes the implementations of the planners that were compared. 

# Installation
For Ubuntu 16.04

1. [Install ROS](http://wiki.ros.org/kinetic/Installation/Ubuntu)

2. Create a catkin workspace, or use an existing one
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
```

3. Install voxblox in the workspace (https://github.com/ethz-asl/voxblox/blob/master/README.md#installation)

4. Get TORQ_GCS code
In src
```
git clone https://github.com/genemerewether/torq.git
```

5. Install Scipy and numpy 
```
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran 
```
Get pip 
```
sudo apt-get install python-pip 
```

Install numpy and scipy: 
```
sudo pip install numpy scipy
```

6. Install dependencies
Install cvx opt dependencis 
```
sudo apt-get install libsuitesparse-dev 
sudo apt-get install libxml2-dev libxslt1-dev
```

7. Install traj
```
cd ~/catkin_ws/src/torq/traj
sudo python setup.py install 
// OR:
sudo python setup.py develop 
```

8. Build px4 messages and torqgcs from catkin
Get into the catkin workspace root (e.g. catkin_ws)
```
cd ~/catkin_ws
catkin build px4_msgs torq_gcs
````


# Startup
Start a ros core
```
roscore
```
Source the workspace
```
cd ~/catkin_ws
source devel/setup.bash
cd src/torq/torq_gcs
```
Run Rviz with the config
```
rviz -d config/race_conops.rviz
```
Start a node
```
python nodes/unco_no_rviz.py
```


# Basic Functionality
Images and more details are [in this googlc document](https://docs.google.com/document/d/1K9KlNsZem-DPdRAYymNsHnRT9RBEyrXbSweh557lFo4/edit?usp=sharing), but may mention data not available here. The basic steps, without images are included below.

1. Load waypoints

- In the RDP tab, click Load Waypoints
- Select file 
  - Choose rand_waypoints_82_001.yaml in (in demo_data)
2.Click Simplify Waypoints
  - This will then initialise the trajectory in the planner - you should see it on the RViz display
3. Manipulate waypoints - Test the capability to change the trajectory with the waypoints
- Move with interactive markers in RVIz
  -Trajectories should update
- Right click markers and delete
- Right click markers and insert
4. Go into the UNCO Tab
5. Optimise time
- Choose a time penalty
- Click optimise
- Wait up to 10-20 seconds
- Trajectory should change, smooth and traj time should change
- Note this will take longer for trajectories with more waypoints
- You can try different weightings to see the effect
6. Scale total time 
-Input desired time
- Click Change Total Time 
- Observe changes - will go faster, but path will be the same
7. Create take-off and landing
8. Click Setup Take-Off and Land
- Observer result
- Will set start position to the origin as default
- Right click waypoints and change type to entry or exit
- Take-off and landing will move
9. Create Laps
- Input number of laps
- Click Close Loop
- Click Create N Laps
10. Animate
- Click checkbox Animate
- Ensure the tf and mesh is being displayed in RViz
11. Save trajectory
- Click Save Trajectory
- It is recommended to load trajectories with the same planner gui as saved

12. Save waypoints
- Click Save Waypoints
- Will just save the waypoints for the given trajectory

13. Check performance
- Click Load Quad Params
- Find and select quad_params_nightwing.yaml (in demo_data)
- In the right table in UNCO, click (Re-calculate) under Max RPM
- The output will be in the table

14. Change Segment time
- Select segment and time under the Change Seg Time tab in the UNCO tab
- Click Change Seg Time
- This will change the time for that specific segment



