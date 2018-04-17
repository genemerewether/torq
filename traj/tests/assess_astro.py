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

class Planner:

    def __init__(self,planner="unco",file_stem="",seed_times=None):

        self.exit_flag = 1 # -1 = fail, 0 = timeout, 1 succeed, -2 input not feasible

        self.planner_type = planner

        self.file_stem = file_stem

        self.app = QApplication( sys.argv )

        global_dict = dict()

        self.disc = torq_gcs.disc.line_simpl.RDPGUI(global_dict)
        self.fsp = torq_gcs.fsp.voxblox_widgets.ESDFGUI(global_dict,new_node=True)

        self.duration = 0

        self.quad_params = None

        self.n_samp = 100

        if self.planner_type is "unco":
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
        if self.planner_type is "unco" or  (self.planner_type is "astro" and not self.test_cylinders):
            filename = self.file_stem+filename[:-5]+"_unco.yaml"
            epsilon = 0.6
        else:
            filename = self.file_stem+filename
            epsilon = 0.0

        self.disc.on_load_waypoints_button_click(False,filename)

        self.disc.on_simplify_button_click(False,epsilon)

        self.plan.on_disc_updated_signal()


        # exit_flag = False
        # while not exit_flag:
        #
        #     # Does this automatically signal?
        #     self.disc.on_simplify_button_click(False,epsilon)
        #
        #     self.plan.on_disc_updated_signal()
        #
        #     if self.planner_type in ["taco","astro_cyl"]:
        #         # Check for collisions
        #         self.plan.on_show_freespace_button_click(True)
        #         l_max = self.plan.compute_nearest_distance_to_obstacles(self.plan.qr_polytraj.waypoints)
        #
        #         if np.min(l_max) > 0.0:
        #             # Feasible path - exit
        #             exit_flag = True
        #             print("Epsilon used is {}".format(epsilon))
        #         else:
        #             # Decrease Epsilon
        #             epsilon = epsilon*0.85
        #
        #     else:
        #         exit_flag = True



    def load_esdf(self,filename):
        self.esdf_file = filename
        self.fsp.on_load_esdf_button_click(False,filename)

    def get_poly(self):

        if self.planner_type in ["astro","astro_cyl"]:
            poly = self.plan.qr_polytraj.create_callable_ppoly()
        else:
            poly = dict()

            for key in self.plan.qr_polytraj.quad_traj.keys():
                poly[key] = self.plan.qr_polytraj.quad_traj[key].piece_poly

        return poly

    """ THIS IS THE MAIN FUNCTION TO PLAN THE TRAJECTORY """
    def plan_trajectory(self,time_penalty = 10,waypoints_file=None,esdf_file=None,params_file=None,seed_times = None):

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


        """ HARD CODED TIME LIMIT HERE """
        time_limit = 5000
        self.exit_flag = 1 # Reset;

        # Start alarm with time limit
        signal.alarm(time_limit)

        if self.planner_type is "unco":
            start_time = time.time()

            samp_mult = 500

            run_obstacles = True

            try:
                while run_obstacles is True:

                    self.plan.qr_polytraj.relative_time_opt(method='COBYLA',options=dict(disp=3,maxiter=1000,tol=0.1))

                    # find the segments in collision
                    collision_segs = self.plan.find_segments_in_collision(samp_mult=samp_mult)

                    if np.size(collision_segs) == 0:
                        run_obstacles = False
                        break

                    # Plan with obstacles
                    self.plan.show_obstacle_plan_updates = False
                    self.plan.on_plan_with_obstacles_button_click()

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

            else:
                try:
                    # Plan with Taco
                    self.plan.qr_polytraj.update_times(self.plan.qr_polytraj.times)
                except TimeoutException:
                    print("Planner {} timed out with waypoints: {}, ESDF: {}".format(self.planner_type,waypoints_file,self.esdf_file))
                    self.exit_flag = 0
                    #continue
                except:
                    print("TACO Failed with Waypoints: {}, ESDF: {}".format(waypoints_file,esdf_file))
                    self.exit_flag = -1
                    #continue

            self.duration = time.time() - start_time

        elif self.planner_type in ["astro","astro_cyl"]:
            start_time = time.time()
            self.plan.run_snap_opt = True

            # import pdb; pdb.set_trace()
            if self.test_cylinders:
                # self.update_markers()

                # if seed_times is not None:
                #     self.replan = False
                #     self.plan.qr_polytraj.update_times(range(len(seed_times)),seed_times)
                # self.update_markers()
                self.replan = True


                # Show free space
                self.plan.on_show_freespace_button_click(True)
                # import pdb; pdb.set_trace()

                self.plan.qr_polytraj.curv_func = False

                self.plan.defer = True
                self.plan.qr_polytraj.feasible = False
                self.plan.qr_polytraj.exit_on_feasible = True
                # initialize corridors
                self.plan.on_initialize_corridors_button_clicked()

                if np.min(self.plan.l_max)<0.0:
                    # Path bad
                    self.exit_flag = -2
                else:
                    # Update weight
                    self.plan.corridor_weight = 1e0
                    self.plan.on_corridor_weight_update_button_clicked()

                    # self.plan.add_esdf_feasibility_checker()

                    self.plan.qr_polytraj.iterations = 0
                    self.plan.replan = True
                    # import pdb; pdb.set_trace()
                    # self.plan.on_run_astro_button_click()
                    while not self.plan.qr_polytraj.feasible:
                    # for i in range(20):
                        import pdb; pdb.set_trace()
                        self.plan.qr_polytraj.optimise(run_one_iteration=True,use_quadratic_line_search=True)#,mutate_serial=3)
                        self.plan.qr_polytraj.get_trajectory()
                        self.update_markers()



            else:

                # ESDF Constraint
                # if seed_times is not None:
                #     self.plan.qr_polytraj.update_times(range(len(seed_times)),seed_times)
                # try:
                    # Run time optimization
                    # converged, iter_count = self.plan.qr_polytraj.time_optimisation(max_iter=1000,run_snap_opt=self.plan.run_snap_opt)
                # Initial plan is on waypoint load
                self.plan.inflate_buffer = 10.0
                self.plan.qr_polytraj.curv_func = False
                self.plan.defer = True
                self.plan.qr_polytraj.feasible = False
                # import pdb; pdb.set_trace()
                # Initialise ESDF obstacle

                self.plan.load_esdf_obstacle(sum_func=True,custom_weighting=True)# This sets exit on feasible to true

                # Set weight
                # weight = 1e-7
                # self.plan.qr_polytraj.set_constraint_weight(weight,"esdf")

                self.plan.qr_polytraj.exit_on_feasible = True

                self.plan.qr_polytraj.iterations = 0

                while not self.plan.qr_polytraj.feasible:
                    import pdb; pdb.set_trace()
                    self.plan.qr_polytraj.optimise(run_one_iteration=True,mutate_serial=3)
                    self.plan.qr_polytraj.get_trajectory()
                    self.update_markers()

                # import pdb; pdb.set_trace()



            #
            # import pdb; pdb.set_trace()
            # self.plan.qr_polytraj.run_astro(replan=True)

                # Plan with obstacles
                # self.plan.qr_polytraj.run_astro_with_increasing_weight()

            # except TimeoutException:
            #     print("Planner {} timed out with waypoints: {}, ESDF: {}".format(self.planner_type,waypoints_file,esdf_file))
            #     self.exit_flag = 0
            #     #continue
            # except:
            #     print("ASTRO Failed with Waypoints: {}, ESDF: {}".format(waypoints_file,esdf_file))
            #     self.exit_flag = -1
            #     #continue


            # Get time
            self.duration = time.time() - start_time

        # Reset alarm
        signal.alarm(0)

        self.update_markers()
        import pdb; pdb.set_trace()

    def update_markers(self):
        print("\n\n\nUpdating path markers\n\n\n")
        if self.planner_type in ["astro","astro_cyl"]:
            self.plan.update_path_markers(waypoints_moved=False)
        else:
            self.plan.update_path_markers()

        if planner_type is not "taco":
            acc_wp = self.plan.get_accel_at_waypoints("main")
            self.plan.interactive_marker_worker.make_controls(self.plan.qr_polytraj.waypoints)
            self.plan.interactive_marker_worker.update_controls(self.plan.qr_polytraj.waypoints,acc_wp = acc_wp)
        # self.plan.obstacle_worker.make_obstacles(self.qr_polytraj.constraint_list)
        # self.plan.obstacle_worker.update_obstacles(self.qr_polytraj.constraint_list)



if __name__ == "__main__":

    """ CHANGE ENVIRONMENT HERE """
    environment = "344"

    if environment is "82":
        esdf_file = '/home/bjm/TORQ/gcs_ws/src/torq_gcs/mesh/82/esdf_building_82_edited.proto'
        waypoints_file = "test_simple.yaml"
    elif environment is "344":
        esdf_file = '/home/bjm/TORQ/gcs_ws/src/torq_gcs/mesh/344_mess/344_mess_esdf.proto'
        wp_set = 3
        waypoints_file = "/home/bjm/TORQ/ompl/waypoints/344_mess/rand_waypoints_344_004.yaml"
    elif environment is "198-B4":
        esdf_file = '/home/bjm/TORQ/gcs_ws/src/torq_gcs/mesh/198-B4/198-B4_esdf.proto'
        waypoints_file = "/home/bjm/TORQ/gcs_ws/src/torq_gcs/waypoints/198-B4/three_wayp_corner_easier.yaml"
        waypoints_file = "/home/bjm/TORQ/gcs_ws/src/torq_gcs/waypoints/198-B4/three_wayp_corner.yaml"
        waypoints_file = "/home/bjm/TORQ/gcs_ws/src/torq_gcs/waypoints/198-B4/three_wayp_corner_challenge_01.yaml"
        # waypoints_file = "/home/bjm/TORQ/ompl/waypoints/198-B4/rand_waypoints_198-B4_V3_001.yaml"
        wp_set = 0
        # waypoints_file = "/home/bjm/TORQ/gcs_ws/src/torq_gcs/waypoints/198-B4/198-B4_3_waypoints.yaml"


    """ CHANGE PLANNER HERE """ # May want to set up to loop through each planner?
    for planner_type in ["astro_cyl"]:#["unco", "taco", "astro"]:
        # planner_type = "unco" # "taco", "astro"

        if planner_type is not "taco":
            # with open("/home/bjm/TORQ/ompl/seg_times_"+environment+"_opt.pickle",'rb') as f:
            with open("/home/bjm/TORQ/ompl/Seg_times/seg_times_"+planner_type+"_"+environment+"_V1_opt.pickle",'rb') as f:
                seg_times = pickle.load(f)
        else:
            seg_times = np.zeros(wp_set+1)
            seg_times[wp_set] = None
        plan = Planner(planner_type,seed_times=seg_times[wp_set])

        plan.load_esdf(esdf_file)

        time_penalty = 1.0

        """ Plan the trajectory """
        try:
            plan.plan_trajectory(time_penalty = time_penalty,waypoints_file=waypoints_file,seed_times=seg_times[wp_set])
        except:
            print("Planner {} failed in environment: {}".format(planner_type,environment))
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
            continue
        # import pdb; pdb.set_trace()
        # """ Save Trajectory File - JUST TO CHECK RESULTS AS DESIRED - save to plot?"""
        # traj_file = waypoints_file[:-5] + "_"+plan.planner_type+".traj"
        # with open(traj_file, 'wb') as f:
        #     print("Saving pickled Traj to {}".format(traj_file))
        #     qr_p_out = plan.plan.qr_polytraj
        #     # Remove esdf constraints
        #     if plan.planner_type is "astro":
        #         qr_p_out.remove_esdf_constraint()
        #
        #     pickle.dump(qr_p_out, f, 2 )

        print("done")

        # """ SAVE DATA-STORE RESULTS - CHANGE FILENAME FOR DIFFERENT PLANNERS and ENVIRONMENTS """
        # save_file = planner_type + "_" + environment + "_data.data"
        # with open(save_file, 'wb') as f:
        #     print("Saving pickled DataStore to {}".format(save_file))
        #     data_out = plan.data_store
        #
        #     pickle.dump(data_out, f, 2 )
