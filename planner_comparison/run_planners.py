# source ~/TORQ/gcs_ws/devel/setup.bash

import sys, os
from python_qt_binding.QtWidgets import QApplication
#from python_qt_binding.QtTest import QTest
import pickle

import numpy
np = numpy
import torq_gcs
from torq_gcs.plan.astro_plan import QRPolyTrajGUI
import time
import pdb, traceback

import diffeo

import signal

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)


# A data structure to store the resutls from batch tests to then save and post process
class DataStore:

    def __init__(self):

        # Dictionaries with a key for each planner
        self.exit_flag = dict()
        self.comp_time = dict()
        self.traj_time = dict()
        self.min_distance = dict()
        self.mean_distance = dict()
        self.feasible = dict()
        self.poly = dict()
        self.waypoints_file = dict()
        self.esdf_file = dict()
        self.esdf_weight = dict()

        self.vel_max  = dict()#(vel_max)
        self.a_max  = dict()#(a_max)
        self.accel_average  = dict()#(accel_average)
        self.snap_average  = dict()#(snap_average)
        self.thr_max  = dict()#(thr_max)
        self.M_max  = dict()#(M_max)
        self.rpm_max  = dict()#(rpm_max)
        self.rpm_min  = dict()#(rpm_min)

        # Initialise as an aray for each planner variant
        for planner_type in ['unco','unco_nothing','taco','astro','astro_cyl']:
            self.exit_flag[planner_type] = []
            self.comp_time[planner_type] = []
            self.traj_time[planner_type] = []
            self.min_distance[planner_type] = []
            self.mean_distance[planner_type] = []
            self.feasible[planner_type] = []
            self.poly[planner_type] = []
            self.waypoints_file[planner_type] = []
            self.esdf_file[planner_type] = []
            self.esdf_weight[planner_type] = []

            self.vel_max[planner_type] = [] #(vel_max)
            self.a_max[planner_type] = [] #(a_max)
            self.accel_average[planner_type] = [] #(accel_average)
            self.snap_average[planner_type] = [] #(snap_average)
            self.thr_max[planner_type] = [] #(thr_max)
            self.M_max[planner_type] = [] #(M_max)
            self.rpm_max[planner_type] = [] #(rpm_max)
            self.rpm_min[planner_type] = [] #(rpm_min)

    def add_data(self,exit_flag,planner_type,comp_time,min_distance,mean_distance,poly,waypoints_file,esdf_file,esdf_weight):
        # To add data from a single run 
        self.exit_flag[planner_type].append(exit_flag)
        self.comp_time[planner_type].append(comp_time)
        self.traj_time[planner_type].append(poly['x'].x[-1])
        self.min_distance[planner_type].append(min_distance)
        self.mean_distance[planner_type].append(mean_distance)
        self.feasible[planner_type].append(min_distance>=0.0)
        self.waypoints_file[planner_type].append(waypoints_file)
        self.esdf_file[planner_type].append(esdf_file)
        self.esdf_weight[planner_type].append(esdf_weight)
        self.poly[planner_type].append(poly)
        print("\n\nAdded data for traj {} for {}, exit flag is {}, min_distance is {}\n\n".format(len(self.exit_flag[planner_type]),planner_type,exit_flag,min_distance))

    def add_performance_data(self,planner_type,vel_max,a_max,accel_average,snap_average,thr_max,M_max,rpm_max,rpm_min):

        self.vel_max[planner_type].append(vel_max)
        self.a_max[planner_type].append(a_max)
        self.accel_average[planner_type].append(accel_average)
        self.snap_average[planner_type].append(snap_average)
        self.thr_max[planner_type].append(thr_max)
        self.M_max[planner_type].append(M_max)
        self.rpm_max[planner_type].append(rpm_max)
        self.rpm_min[planner_type].append(rpm_min)


""" 
Quickly made planner class emulating the TORQ GUI widgets for the planners

Uses the same framework as the GUI to initialise the planners and then creates a 
number of utility functions around it for the batch testing and variations with different planners. 

"""
class Planner:

    def __init__(self,planner="unco",file_stem="",seed_times=None):


        ## Iniit settings
        self.exit_flag = 1 # -1 = fail, 0 = timeout, 1 succeed, -2 input not feasible

        self.planner_type = planner

        self.file_stem = file_stem

        self.app = QApplication( sys.argv )

        global_dict = dict()

        self.disc = torq_gcs.disc.line_simpl.RDPGUI(global_dict)
        self.fsp = torq_gcs.fsp.voxblox_widgets.ESDFGUI(global_dict,new_node=True)

        self.data_store = DataStore()

        self.duration = 0

        self.quad_params = None

        self.n_samp = 100

        # Change initialisation based on planner type
        if self.planner_type in ["unco","unco_nothing"]:
            self.plan = torq_gcs.plan.unco.QRPolyTrajGUI(global_dict)#, new_node=True)
        elif self.planner_type is "taco":
            self.plan = torq_gcs.plan.taco.FreespacePlannerGUI(global_dict)
        elif self.planner_type is "astro":
            self.plan = torq_gcs.plan.astro_plan.QRPolyTrajGUI(global_dict,defer=False,curv_func=False)
            self.test_cylinders = False
            self.plan.seed_times = seed_times
        elif self.planner_type is "astro_cyl":
            self.plan = torq_gcs.plan.astro_plan.QRPolyTrajGUI(global_dict,defer=False,curv_func=False)
            self.test_cylinders = True
            self.plan.seed_times = seed_times

    def load_waypoints(self,filename):
        self.waypoints_file = filename

        # UNCO has different files to load (Dense with waypoints)
        if self.planner_type is "unco" or  self.planner_type is "astro":
            filename = self.file_stem+filename[:-5]+"_unco.yaml"
            epsilon = 0.6
        # elif self.planner_type is "astro":
        #     filename = self.file_stem+filename
        #     epsilon = 0.6
        else:
            filename = self.file_stem+filename
            epsilon = 0.0

        # Use the GU path to load waypoints
        self.disc.on_load_waypoints_button_click(False,filename)

        # Signal to get callback
        self.disc.on_simplify_button_click(False,epsilon)

        # Execute callback to load waypoints into planner
        self.plan.on_disc_updated_signal()


    def load_esdf(self,filename):
        self.esdf_file = filename
        self.fsp.on_load_esdf_button_click(False,filename)

    def load_quad_params(self,filename):
        with open(filename, 'rb') as f:
            self.quad_params = diffeo.controls.load_params(filename)

    """
    Process a trajectory from a test to get the performance metrics (accel, RPM etc.)
    Uses the differential flatness transformation
    """
    def get_performance_metrics(self,poly=None):
        
        if self.quad_params is None:
            return

        # Get poly
        if poly is None:
            poly = self.get_poly()

        t_vec = np.linspace(poly['x'].x[0],poly['x'].x[-1],self.n_samp/2*poly['x'].x[-1])

        a_max = 0.0
        rpm_max = 0.0
        rpm_min = 999999990.0
        thr_max = 0.0
        vel_max = 0.0
        M_max = 0.0

        snap_sum = 0.0
        accel_sum = 0.0

        vel = np.array([poly['x'].derivative()(t_vec),
        poly['y'].derivative()(t_vec),
        poly['z'].derivative()(t_vec)])

        accel = np.array([poly['x'].derivative().derivative()(t_vec),
        poly['y'].derivative().derivative()(t_vec),
        poly['z'].derivative().derivative()(t_vec)])

        jerk = np.array([poly['x'].derivative().derivative().derivative()(t_vec),
        poly['y'].derivative().derivative().derivative()(t_vec),
        poly['z'].derivative().derivative().derivative()(t_vec)])

        snap = np.array([poly['x'].derivative().derivative().derivative().derivative()(t_vec),
        poly['y'].derivative().derivative().derivative().derivative()(t_vec),
        poly['z'].derivative().derivative().derivative().derivative()(t_vec)])

        yaw = poly['yaw'](t_vec)
        yaw_dot = poly['yaw'].derivative()(t_vec)
        yaw_ddot = poly['yaw'].derivative().derivative()(t_vec)

        for i in range(len(t_vec)):
            v_mag = np.linalg.norm(vel[:,i])
            a_mag = np.linalg.norm(accel[:,i])

            accel_sum += a_mag
            snap_sum += np.linalg.norm(snap[:,i])

            # Get rotation matrix
            R, data = diffeo.body_frame.body_frame_from_yaw_and_accel(yaw[i], accel[:,i],'matrix')

            # Thrust
            thrust, thrust_mag = diffeo.body_frame.get_thrust(accel[:,i])

            # Angular rates
            ang_vel = diffeo.angular_rates_accel.get_angular_vel(thrust_mag,jerk[:,i],R,yaw_dot[i])

            # Angular accelerations
            ang_accel = diffeo.angular_rates_accel.get_angular_accel(thrust_mag,jerk[:,i],snap[:,i],R,ang_vel,yaw_ddot[i])

            # torques
            torques = diffeo.controls.get_torques(ang_vel, ang_accel, self.quad_params)

            # Get rotor speeds
            rpm = diffeo.controls.get_rotor_speeds(torques,thrust_mag*self.quad_params['mass'],self.quad_params)

            max_torque = np.max(torques)
            max_rpm = np.max(rpm)
            min_rpm = np.min(rpm)

            # Update maximums
            if thrust_mag > thr_max:
                thr_max = thrust_mag
            if a_mag > a_max:
                a_max = np.linalg.norm(accel)
            if max_rpm > rpm_max:
                rpm_max = max_rpm
            if min_rpm > rpm_min:
                rpm_min = min_rpm
            if max_torque > M_max:
                M_max = max_torque
            if v_mag > vel_max:
                vel_max = v_mag

        accel_average = accel_sum/len(t_vec)
        snap_average = snap_sum/len(t_vec)

        # Add to data store structure
        self.data_store.add_performance_data(self.planner_type,vel_max,a_max,accel_average,snap_average,thr_max,M_max,rpm_max,rpm_min)

    """
    Store the data from the trajectory
    """
    def store_plan_data(self):

        # Get poly
        poly = self.get_poly()

        dist = self.plan.check_distance_to_trajectory(samp_mult=self.n_samp*5)

        mean_distance = np.linalg.norm(dist)

        min_distance = np.min(dist)


        # self.get_performance_metrics(poly)

        if self.planner_type is ["astro"]:#,"astro_cyl"]:
            esdf_weight = self.plan.qr_polytraj.constraint_list[0].weight
        else:
            esdf_weight = 0

        self.data_store.add_data(self.exit_flag,self.planner_type,self.duration,min_distance,
        mean_distance,poly,self.waypoints_file,self.esdf_file,esdf_weight)

    """
    Get the polynomial for a trajectory to analyse its characteristics
    """
    def get_poly(self):

        if self.planner_type in ["astro","astro_cyl"]:
            poly = self.plan.qr_polytraj.create_callable_ppoly()
        else:
            poly = dict()

            for key in self.plan.qr_polytraj.quad_traj.keys():
                poly[key] = self.plan.qr_polytraj.quad_traj[key].piece_poly

        return poly

    """ THIS IS THE MAIN FUNCTION TO PLAN THE TRAJECTORY """
    def plan_trajectory(self,time_penalty = 10,waypoints_file=None,esdf_file=None,params_file=None,seed_times=None):

        # Set time penalty
        if self.planner_type is not "taco":
            self.plan.on_time_penalty_line_edit_text_edit(time_penalty)

        # load waypoints
        if waypoints_file is not None:
            self.waypoints_file = waypoints_file
            self.load_waypoints(waypoints_file)

        # Load esdf
        if esdf_file is not None:
            self.esdf_file = esdf_file
            self.load_esdf(esdf_file)

        if params_file is not None:
            self.params_file = params_file
            self.load_quad_params(params_file)

        """ HARD CODED TIME LIMIT HERE """
        time_limit = 50

        self.exit_flag = 1 # Reset;

        # Start alarm with time limit
        signal.alarm(time_limit)

        if self.planner_type in ["unco","unco_nothing"]:
            start_time = time.time()

            samp_mult = 500

            run_obstacles = True


            try:
                # Set seed times if input
                if seed_times is not None:
                    self.plan.qr_polytraj.update_times(seed_times)

                if self.planner_type is "unco_nothing":
                    run_obstacles = False

                while run_obstacles is True:

                    # find the segments in collision
                    collision_segs = self.plan.find_segments_in_collision(samp_mult=samp_mult)

                    if np.size(collision_segs) == 0:
                        run_obstacles = False
                        break

                    # Plan with obstacles
                    self.plan.show_obstacle_plan_updates = False
                    self.plan.on_plan_with_obstacles_button_click()
                # import pdb; pdb.set_trace()
            except TimeoutException:
                print("Planner {} timed out with waypoints: {}, ESDF: {}".format(self.planner_type,waypoints_file,esdf_file))
                self.exit_flag = 0
                #continue
            except:
                print("UNCO Failed with waypoints: {}, ESDF: {}".format(waypoints_file,esdf_file))
                self.exit_flag = -1
                #continue

            self.duration = time.time() - start_time

        elif self.planner_type is "taco":
            start_time = time.time()

            samp_mult = 500

            self.plan.qr_polytraj.restrict_freespace = True

            # Set A_max
            self.plan.on_A_max_line_edit_text_edit(20.0)

            # Plan with obstacles
            if np.sum(self.plan.l_max<=0.0) > 0:
                # Path in collision -
                print("Initial Path is not free from collisions. Waypoints: {}, ESDF: {}".format(waypoints_file,self.esdf_file))
                self.exit_flag = -2
                # import pdb; pdb.set_trace()

            else:

                try:
                    # Plan with Taco
                    self.plan.qr_polytraj.update_times(self.plan.qr_polytraj.times)
                except TimeoutException:
                    print("Planner {} timed out with waypoints: {}, ESDF: {}".format(self.planner_type,waypoints_file,self.esdf_file))
                    self.exit_flag = 0
    
                except:
                    print("TACO Failed with Waypoints: {}, ESDF: {}".format(waypoints_file,esdf_file))
                    self.exit_flag = -1
    

            self.duration = time.time() - start_time

        elif self.planner_type in ["astro","astro_cyl"]:
            start_time = time.time()

            try:

                if self.test_cylinders:

                    self.plan.qr_polytraj.curv_func = False
                    self.plan.defer = True
                    self.plan.qr_polytraj.feasible = False
                    self.plan.qr_polytraj.exit_on_feasible = True
                    # initialize corridors
                    self.plan.inflate_buffer = 10

                    self.plan.on_initialize_corridors_button_clicked(sum_func=False)

                    if np.min(self.plan.l_max)<0.0:
                        # Path bad
                        self.exit_flag = -2
                    else:


                        self.plan.add_esdf_feasibility_checker()# This sets exit on feasible to true

                        self.plan.qr_polytraj.exit_on_feasible = True

                        self.plan.replan = True

                        check_seed_violation = False

                        if check_seed_violation:
                            self.plan.qr_polytraj.total_cost_grad_curv(self.plan.qr_polytraj.c_leg_poly,doGrad=False)

                            if self.plan.qr_polytraj.feasible:
                                self.exit_flag = 2
                            else:
                                self.exit_flag = -3


                        else:
                            # # Update weight
                            self.plan.corridor_weight = 1e0#-4
                            self.plan.on_corridor_weight_update_button_clicked()

                            self.plan.qr_polytraj.optimise(use_quadratic_line_search=True)

                            self.plan.qr_polytraj.get_trajectory()

                        self.plan.defer = False
                        # self.plan.qr_polytraj.curv_func = True
                        self.plan.qr_polytraj.exit_on_feasible = False
                        self.plan.replan = False


                else:
                    
                    # Initialise ESDF obstacle
                    self.plan.inflate_buffer = 10.0
                    self.plan.qr_polytraj.curv_func = False
                    self.plan.qr_polytraj.feasible = False
                    self.plan.defer = True
                   
                    self.plan.load_esdf_obstacle(sum_func=True,custom_weighting=True)# This sets exit on feasible to true

                    self.plan.qr_polytraj.exit_on_feasible = True

                    self.plan.qr_polytraj.optimise(mutate_serial=3)
                    self.plan.qr_polytraj.get_trajectory()
                    self.plan.defer = False
                    self.plan.qr_polytraj.curv_func = True

                    if self.plan.qr_polytraj.mutated:
                        # Flag to track how often it mutated
                        self.exit_flag = 2

                    """
                    OPTIONS BELOW TO HAVE A FIXED WEIGHTING FOR THE OBSTACLES RATHER THAN CUSTOM
                    """
                    # Set weight
                    # weight = 1e3
                    # self.plan.qr_polytraj.set_constraint_weight(weight,"esdf")
                    #
                    # import pdb; pdb.set_trace()
                    # self.plan.qr_polytraj.run_astro(replan=True)

                        # Plan with obstacles
                        # self.plan.qr_polytraj.run_astro_with_increasing_weight()

            except TimeoutException:
                print("\n\nPlanner {} timed out with waypoints: {}, ESDF: {}\n\n".format(self.planner_type,waypoints_file,self.esdf_file))
                self.exit_flag = 0

                #continue
            except:
                print("\n\nASTRO Failed with Waypoints: {}, ESDF: {}\n\n".format(waypoints_file,esdf_file))
                self.exit_flag = -1
                # Reset alarm
                signal.alarm(0)
                

                


            # Get time
            self.duration = time.time() - start_time

            if self.planner_type is "astro_cyl":
                self.plan.qr_polytraj.remove_corridor_constraints()

        # Reset alarm
        signal.alarm(0)

        self.store_plan_data()


    """ 
    FUNCTION FOR GENERATING OPTIMAL SEGMENT TIMES

    Run once to get for ASTRO and UNCO, then just load the optimal segment times for later batch runs. 

    """
    def generate_optimal_seg_times(self,waypoints_file,planner_type = "astro",time_penalty = 1):

        # Update the time penalty for the time optimisation
        self.plan.on_time_penalty_line_edit_text_edit(time_penalty)

        # Process waypoint name for different planners
        if waypoints_file is None:
            error("Need to input a waypoints file")
        if planner_type is "unco":
            epsilon = 0.6
            filename = self.file_stem+waypoints_file[:-5]+"_unco.yaml"
        elif planner_type is "astro":
            epsilon = 0.6
            filename = self.file_stem+waypoints_file[:-5]+"_unco.yaml"
        else:
            filename = self.file_stem+waypoints_file
            epsilon = 0.0

        # Load the waypoints
        self.disc.on_load_waypoints_button_click(False,filename)
        self.disc.on_simplify_button_click(False,epsilon)
        self.plan.on_disc_updated_signal()

        # Time optimisation
        start_time = time.time()
        self.plan.qr_polytraj.relative_time_opt(method='COBYLA',options=dict(disp=3,maxiter=1000,tol=0.1))
        time_opt_timer = time.time() - start_time

        # CHeck the trajectory for collisions
        dist = self.plan.check_distance_to_trajectory(samp_mult=self.n_samp*5)

        min_distance = np.min(dist)

        return time_opt_timer, min_distance


if __name__ == "__main__":

    """ CHANGE ENVIRONMENT HERE (see list below) """
    environment = "344"

    """
    Filepaths are defined below - THESE WILL NEED TO BE CHANGED WHEN RUNNNG IT ELSEWHERE
    """
    if environment is "344":
        esdf_file = 'ESDFs/BatchSimulations/LargeWarehouse/344_mess_esdf.proto'
        waypoints_file_root = "rand_waypoints_344_"
        waypoints_file_stem = "Waypoints/BatchSimulations/LargeWarehouse/"
    elif environment is "198-B4":
        esdf_file = 'ESDFs/BatchSimulations/SmallLab/198-B4_esdf.proto'
        waypoints_file_root = "rand_waypoints_198-B4_V3_"
        waypoints_file_stem = "Waypoints/BatchSimulations/SmallLab/'

    # Parameter file for computing the perofrmance metrics (FILEPATH WILL NEED TO BE UPDATED)
    params_file = 'quad_params_nightwing.yaml'

    # Change this flag to true to generate the segment times
    generate_seg_times = False


    if generate_seg_times:
        time_opt_timer = []
        seg_times = []
        dist_store = []


    """ CHANGE PLANNER(S) HERE """ 
    # Will loop through each planner
    for planner_type in ["unco","taco","astro","astro_cyl"]:
        
        if not generate_seg_times:
            # with open("seg_times_"+planner_type+"_"+environment+"_opt.pickle",'rb') as f:
            if planner_type is not "taco":
              if environment is "344":
                with open("SegTimes/seg_times_"+planner_type+"_"+environment+"_V3_opt.pickle",'rb') as f:
                  seg_times = pickle.load(f)
              elif environment is "198-B4":
                with open("SegTimes/seg_times_"+planner_type+"_"+environment+"_V1_opt.pickle",'rb') as f:
                    seg_times = pickle.load(f)
            else:
                seg_times = [[0]]*100

        plan = []
        if generate_seg_times:
            # Initialise planner with UNCO for time optimisation
            plan = Planner("unco",waypoints_file_stem)

            plan.load_esdf(esdf_file)
        else:
            # Initialise planner 
            plan = Planner(planner_type,waypoints_file_stem,seed_times = seg_times[0])

            plan.load_esdf(esdf_file)
            plan.load_quad_params(params_file)

        """ CHANGE THE NUMBER OF TRAJECTORIES TO LOOP THROUGH HERE """ # Match to number of waypoint sets generated with OMPL
        n_traj = 10# 1 to test, 100 ideally

        # time penalty setting dependent on the planner
        if planner_type is "unco":
            time_penalty = 1.0
        elif planner_type is "astro":
            time_penalty = 1.0
        else:
            time_penalty = 1.0


        # Loops through each trajectory
        for i in range(n_traj):

            # Get filename
            if i < 9:
                waypoints_file = waypoints_file_root + "00" + str(i+1) + ".yaml"
            elif i < 99:
                waypoints_file = waypoints_file_root + "0" + str(i+1) + ".yaml"
            else:
                waypoints_file = waypoints_file_root + str(i+1) + ".yaml"


            if generate_seg_times:
                # Generate the optimal time 
                opt_time, min_distance = plan.generate_optimal_seg_times(waypoints_file,planner_type = planner_type,time_penalty = time_penalty)
                time_opt_timer.appe0nd(opt_time)
                seg_times.append(plan.plan.qr_polytraj.times)
                dist_store.append(min_distance)
                if min_distance < 0.0:
                    print("\nIn collision!!\n")
                else:
                    print("\nNot in collision.\n")
            else:
                """ Plan the trajectory """
                try:
                    if planner_type in ["astro","astro_cyl"]:
                        plan.plan.seed_times = seg_times[i]
                    plan.plan_trajectory(time_penalty = time_penalty,waypoints_file=waypoints_file,seed_times = seg_times[i])
                except:
                    print("Planner {} failed on trajectory {} in environment: {}".format(planner_type,i+1,environment))
                    # type, value, tb = sys.exc_info()
                    # traceback.print_exc()
                    # pdb.post_mortem(tb)
                    continue



                # UNCOMMENT BELOW IF IT IS DESIRED TO SAVE TRAJECTORIES
                # if not generate_seg_times:
                #     """ Save Trajectory File - JUST TO CHECK RESULTS AS DESIRED - save to plot?"""
                #     traj_file = waypoints_file[:-5] + "_"+plan.planner_type+"_new.traj"
                #     with open(traj_file, 'wb') as f:
                #         print("Saving pickled Traj to {}".format(traj_file))
                #         qr_p_out = plan.plan.qr_polytraj
                #         # Remove esdf constraints
                #         if planner_type is "astro":
                #             qr_p_out.remove_esdf_constraint()
                #         if planner_type is "astro_cyl":
                #             if qr_p_out.esdf_feasibility_check:
                #                 qr_p_out.remove_esdf_constraint()

                #         pickle.dump(qr_p_out, f, 2 )

        print("done")

        if generate_seg_times:
            # Save the optimal segment times and the time to compute them
            print("Number in collision is {} out of {}".format(np.sum(np.array(dist_store)<0.0),n_traj))
            with open("seg_times_"+planner_type+"_"+environment+"_V1_opt.pickle",'wb') as f:
                pickle.dump(seg_times,f,2)
            with open("time_opt_times_"+planner_type+"_"+environment+"_V1_opt.pickle",'wb') as f:
                pickle.dump(time_opt_timer,f,2)
            with open("dist_store_"+planner_type+"_"+environment+"_V1_opt.pickle",'wb') as f:
                pickle.dump(dist_store,f,2)

        else:
            """ SAVE DATA-STORE RESULTS - CHANGE FILENAME FOR DIFFERENT PLANNERS and ENVIRONMENTS """
            save_file = "Results/"+planner_type + "_" + environment + "_V3_esdf_fix_max_data.data"
            with open(save_file, 'wb') as f:
                print("Saving pickled DataStore to {}".format(save_file))
                data_out = plan.data_store

                pickle.dump(data_out, f, 2 )

