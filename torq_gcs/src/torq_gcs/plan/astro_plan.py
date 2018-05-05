"""
Copyright 2016, by the California Institute of Technology. ALL RIGHTS
RESERVED. United States Government Sponsorship acknowledged. Any commercial
use must be negotiated with the Office of Technology Transfer at the
California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this
software, the user agrees to comply with all applicable U.S. export laws and
regulations. User has the responsibility to obtain export licenses, or other
export authority as may be required before exporting such information to
foreign countries or providing access to foreign persons.
"""

import sys
import os
import pickle
import time

import numpy
np = numpy

import astro
from astro import traj_qr
from astro import constraint
from astro import utils
from diffeo import body_frame

import rospkg, rospy, tf
from python_qt_binding import loadUi
from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *
from python_qt_binding.QtWidgets import *

import torq_gcs
from torq_gcs.plan.ros_helpers import TrajectoryDisplayWorker
from torq_gcs.plan.ros_helpers import WaypointControlWorker
from torq_gcs.plan.ros_helpers import ObstacleControlWorker
from torq_gcs.plan.ros_helpers import NodePathWorker
from torq_gcs.plan.ros_helpers import AnimateWorker

from px4_msgs.msg import PolyTraj as PolyTraj_msg
from geometry_msgs.msg import PoseStamped

import transforms3d
from transforms3d import quaternions

class OptimizerWorker(QObject):
    finished = Signal() # class variable shared by all instances
    def __init__(self, finished_callback, parent=None):
        super(OptimizerWorker, self).__init__(parent)
        self.finished.connect(finished_callback)
        self.is_running = True

    def task(self):
        print("Starting Optimisation.\n Times are: {}, run snap opt is {}".format(self.local_qr_polytraj.times,self.run_snap_opt))
        converged = False

        while not converged and self.is_running:
            converged, iter_count = self.local_qr_polytraj.time_optimisation(max_iter=3000,run_snap_opt=self.run_snap_opt)
            print("Finished Optimisation in {} iterations\nTimes are: {}".format(iter_count,self.local_qr_polytraj.times))
            # TODO (bmorrell@jpl.nasa.gov) Implement stopping capability to the optimisation
            # Maybe set it for 1 iteration limit?
            print("Normalised time ratios are: {}".format(self.local_qr_polytraj.times/np.max(self.local_qr_polytraj.times)))
            self.local_qr_polytraj.run_astro(replan=True)

        self.finished.emit()

    def stop(self):
        self.is_running = False

class QRPolyTrajGUI(QWidget):
    def __init__(self,
                 global_dict,
                 node_name='astro_plan',
                 new_node=False,
                 parent=None,
                 defer=False,
                 curv_func=False):
        super(QRPolyTrajGUI, self).__init__(parent)

        ui_file = os.path.join(rospkg.RosPack().get_path('torq_gcs'),
                               'resource', 'plan', 'Astro.ui')
        loadUi(ui_file, self)

        self.qr_polytraj = None
        self.qr_p_entry = None
        self.qr_p_exit = None
        self.time_penalty = 1e5
        self.ppoly_laps = None

        self.curv_func = curv_func

        self.entry_ID = 0
        self.exit_ID = None
        self.n_laps = 1
        self.entry_time = 4.0
        self.exit_time = 4.0
        self.laps_set_flag = False # indicates whether the laps/entry and exit have been set

        self.time_edit_index = 0
        self.edit_time = 1.0
        self.total_time_setting = None

        self.last_pose_time   = None
        self.last_position    = None
        self.last_orientation = None
        self.controls_enabled = True
        self.run_snap_opt = False
        self.new_x_waypoint = 0.
        self.new_y_waypoint = 0.
        self.new_z_waypoint = 0.
        self.new_yaw_waypoint = 0.
        self.index_waypoint = 0
        self.collide_sphere_dist = 0.0
        self.constraint_list = []
        self.accel_lim = 1e2
        self.accel_weight = 1e0
        self.corridor_weight = 1e-4
        self.all_corridors = True # IF false, then will only add corridors for segments in violation
        self.esdf_weight = 100.0
        self.nurbs_weight = 100.0
        self.replan = True
        
        self.seed_times = None

        self.defer = defer

        # Settings for showing distance to obstacles
        self.unknown_is_free = True
        self.quad_buffer = 0.3
        self.inflate_buffer = 10.0

         #15
        self.l_max = None
        self.plot_traj_color = True
        self.plot_path_color = True
        self.corridors_init = False

        if new_node:
            rospy.init_node(node_name, anonymous=True)
        # NOTE (mereweth@jpl.nasa.gov) - this still does not bypass the GIL
        # unless rospy disables the GIL in C/C++ code
        self.ros_helper_thread = QThread()
        self.ros_helper_thread.app = self
        self.marker_worker = TrajectoryDisplayWorker(
                                            frame_id="local_origin",
                                            marker_topic="trajectory_marker",
                                            qr_type="main")
        self.marker_worker.moveToThread(self.ros_helper_thread)

        self.marker_path_worker = NodePathWorker(
                                            frame_id="local_origin",
                                            marker_topic="path_marker")
        self.marker_path_worker.moveToThread(self.ros_helper_thread)

        self.interactive_marker_worker = WaypointControlWorker(
                                            self.on_waypoint_control_callback,
                                            menu_callback=self.on_waypoint_menu_control_callback,
                                            frame_id="local_origin",
                                            marker_topic="trajectory_control",
                                            qr_type="main")
        self.interactive_marker_worker.moveToThread(self.ros_helper_thread)

        self.obstacle_worker = ObstacleControlWorker(
                                            self.on_obstacle_control_callback,
                                            frame_id="local_origin",
                                            marker_topic="obstacle_control")
        self.obstacle_worker.moveToThread(self.ros_helper_thread)

        self.marker_worker_entry = TrajectoryDisplayWorker(
                                            frame_id="local_origin",
                                            marker_topic="entry_marker",
                                            qr_type="entry")
        self.marker_worker_entry.moveToThread(self.ros_helper_thread)

        self.marker_worker_exit = TrajectoryDisplayWorker(
                                            frame_id="local_origin",
                                            marker_topic="exit_marker",
                                            qr_type="exit")
        self.marker_worker_exit.moveToThread(self.ros_helper_thread)

        self.interactive_marker_worker_entry = WaypointControlWorker(
                                            self.on_waypoint_control_callback,
                                            menu_callback=self.on_waypoint_menu_control_callback,
                                            frame_id="local_origin",
                                            marker_topic="entry_control",
                                            qr_type="entry")
        self.interactive_marker_worker_entry.moveToThread(self.ros_helper_thread)

        self.interactive_marker_worker_exit = WaypointControlWorker(
                                            self.on_waypoint_control_callback,
                                            menu_callback=self.on_waypoint_menu_control_callback,
                                            frame_id="local_origin",
                                            marker_topic="exit_control",
                                            qr_type="exit")
        self.interactive_marker_worker_exit.moveToThread(self.ros_helper_thread)


        self.animate_worker = AnimateWorker(self.on_animate_eval_callback,
                                            frame_id="minsnap_animation",
                                            parent_frame_id='local_origin',
                                            marker_topic="minsnap_animation_marker",
                                            slowdown=1)
        self.animate_worker.moveToThread(self.ros_helper_thread)

        self.ros_helper_thread.start()

        self.polytraj_pub = rospy.Publisher('trajectory', PolyTraj_msg,
                                            queue_size=1)
        self.pose_sub = rospy.Subscriber("pose_stamped_out", PoseStamped,
                                         self.callback_pose_stamped)
        # self.setpoint_pub = rospy.Publisher("setpoint_unreal",PoseStamped, queue_size = 1)
        # self.setpointCount = 0

        self.global_dict = global_dict

        self.animate_checkbox.toggled.connect(self.on_animate_radio_button_toggle)
        self.send_trajectory_button.clicked.connect(self.on_send_trajectory_button_click)
        #
        self.load_trajectory_button.clicked.connect(self.on_load_trajectory_button_click)
        self.save_trajectory_button.clicked.connect(self.on_save_trajectory_button_click)
        self.optimize_button.clicked.connect(self.on_optimize_button_click)
        self.stop_optimize_button.clicked.connect(self.on_stop_optimize_button_click)
        self.mutate_astro_button.clicked.connect(self.on_mutate_optimize_button_click)
        self.set_first_waypoint_to_drone_button.clicked.connect(self.on_set_first_waypoint_to_drone_button_click)
        self.set_first_waypoint_above_drone_button.clicked.connect(self.on_set_first_waypoint_above_drone_button_click)

        self.set_start_and_end.clicked.connect(self.on_set_takeoff_and_landing_to_drone_click)
        self.add_takeoff_button.clicked.connect(self.on_add_takeoff_button_click)
        self.add_landing_button.clicked.connect(self.on_add_landing_button_click)

        self.save_waypoints_from_traj_button.clicked.connect(self.on_save_waypoints_from_traj_click)

        self.set_yaw_to_traj_button.clicked.connect(self.on_set_yaw_to_traj_button_click)
        self.set_yaw_to_zero_button.clicked.connect(self.on_set_yaw_to_zero_button_click)
        # self.create_n_laps_button.clicked.connect(self.on_create_n_laps_button_click)

        self.run_snap_opt_radio_button.clicked.connect(self.on_run_snap_opt_radio_button_click)
        self.add_waypoint_button.clicked.connect(self.on_add_waypoint_button_click)
        self.delete_waypoint_button.clicked.connect(self.on_delete_waypoint_button_click)

        self.check_distance_button.clicked.connect(self.on_check_distance_button_click)
        self.show_collision_checkbox.toggled.connect(self.on_show_collision_checkbox_toggled)
        self.show_freespace_checkbox.toggled.connect(self.on_show_freespace_button_click)

        self.load_obstacles_button.clicked.connect(self.on_load_obstacles_button_click)
        self.init_esdf_obstacle_button.clicked.connect(self.load_esdf_obstacle)
        self.create_two_waypoints_button.clicked.connect(self.create_two_waypoints)
        self.run_astro_button.clicked.connect(self.on_run_astro_button_click)
        self.run_astro_with_weight_growth_button.clicked.connect(self.on_run_astro_with_weight_growth_button_click)
        self.add_obstacle_button.clicked.connect(self.on_add_ellipsoid_constraint_click)

        self.initialize_corridors_button.clicked.connect(self.on_initialize_corridors_button_clicked)
        self.corridor_weight_update_button.clicked.connect(self.on_corridor_weight_update_button_clicked)
        self.esdf_weight_update_button.clicked.connect(self.on_esdf_weight_update_button_clicked)

        self.accel_limit_button.clicked.connect(self.update_accel_limit)

        self.change_segment_time.clicked.connect(self.on_update_times_button_clicked)
        self.change_total_time.clicked.connect(self.on_scale_total_time_button_clicked)

        double_validator = QDoubleValidator(parent=self.time_penalty_line_edit)
        double_validator.setBottom(0)
        self.time_penalty_line_edit.setValidator(double_validator)
        self.time_penalty_line_edit.setText(str(self.time_penalty))
        self.time_penalty_line_edit.textEdited.connect(self.on_time_penalty_line_edit_text_edit)

        self.traj_dist_line_edit.setText("N/A")

        self.comp_time_line_edit.setValidator(double_validator)

        self.outer_time_line_edit.setValidator(double_validator)

        self.traj_time_line_edit.setValidator(double_validator)
        self.traj_accel_line_edit.setValidator(double_validator)

        self.accel_lim_line_edit.setValidator(double_validator)
        self.accel_lim_line_edit.setText(str(self.accel_lim))
        self.accel_lim_line_edit.textEdited.connect(self.on_accel_lim_line_edit_text_edit)

        self.accel_weight_line_edit.setValidator(double_validator)
        self.accel_weight_line_edit.setText(str(self.accel_weight))
        self.accel_weight_line_edit.textEdited.connect(self.on_accel_weight_line_edit_text_edit)

        self.corridor_weight_line_edit.setValidator(double_validator)
        self.corridor_weight_line_edit.setText(str(self.corridor_weight))
        self.corridor_weight_line_edit.textEdited.connect(self.on_corridor_weight_line_edit_text_edit)

        self.esdf_weight_line_edit.setValidator(double_validator)
        self.esdf_weight_line_edit.setText(str(self.esdf_weight))
        self.esdf_weight_line_edit.textEdited.connect(self.on_esdf_weight_line_edit_text_edit)

        double_validator_wayp = QDoubleValidator(parent=self.x_waypoint_line_edit)
        double_validator_wayp.setBottom(-100.0)
        double_validator_wayp.setTop(100.0)
        self.x_waypoint_line_edit.setValidator(double_validator_wayp)
        self.x_waypoint_line_edit.setText("0")
        self.x_waypoint_line_edit.textEdited.connect(self.on_x_waypoint_line_edit_text_edit)
        self.y_waypoint_line_edit.setValidator(double_validator_wayp)
        self.y_waypoint_line_edit.setText("0")
        self.y_waypoint_line_edit.textEdited.connect(self.on_y_waypoint_line_edit_text_edit)
        self.z_waypoint_line_edit.setValidator(double_validator_wayp)
        self.z_waypoint_line_edit.setText("0")
        self.z_waypoint_line_edit.textEdited.connect(self.on_z_waypoint_line_edit_text_edit)
        self.yaw_waypoint_line_edit.setValidator(double_validator_wayp)
        self.yaw_waypoint_line_edit.setText("0")
        self.yaw_waypoint_line_edit.textEdited.connect(self.on_yaw_waypoint_line_edit_text_edit)
        self.collide_sphere_val_line_edit.setValidator(double_validator_wayp)
        self.collide_sphere_val_line_edit.setText("0")
        self.collide_sphere_val_line_edit.textEdited.connect(self.on_collide_sphere_val_line_edit_text_edit)
        int_validator = QIntValidator(0,200,parent=self.index_line_edit)
        self.index_line_edit.setValidator(int_validator)
        self.index_line_edit.setText("0")
        self.index_line_edit.textEdited.connect(self.on_index_line_edit_text_edit)

        self.iteration_line_edit.setValidator(int_validator)

        # int_validator_n_laps = QIntValidator(0,200,parent=self.n_laps_line_edit)
        # self.n_laps_line_edit.setValidator(int_validator_idx)
        # self.n_laps_line_edit.setText("1")
        # self.n_laps_line_edit.textEdited.connect(self.on_n_laps_line_edit_text_edit)

        # double_validator_entry_time = QDoubleValidator(parent=self.entry_time_line_edit)
        self.entry_time_line_edit.setValidator(double_validator_wayp)
        self.entry_time_line_edit.setText(str(self.entry_time))
        self.entry_time_line_edit.textEdited.connect(self.on_entry_time_line_edit_text_edit)

        # double_validator_exit_time = QDoubleValidator(parent=self.exit_time_line_edit)
        self.exit_time_line_edit.setValidator(double_validator_wayp)
        self.exit_time_line_edit.setText(str(self.exit_time))
        self.exit_time_line_edit.textEdited.connect(self.on_exit_time_line_edit_text_edit)

        self.time_index_line_edit.setValidator(int_validator)
        self.time_index_line_edit.setText("0")
        self.time_index_line_edit.textEdited.connect(self.on_time_index_line_edit_text_edit)

        self.edit_time_line_edit.setValidator(double_validator_wayp)
        self.edit_time_line_edit.setText(str(self.edit_time))
        self.edit_time_line_edit.textEdited.connect(self.on_edit_time_line_edit_text_edit)

        self.total_time_line_edit.setValidator(double_validator_wayp)
        # self.edit_time_line_edit.setText(str(self.edit_time))
        self.total_time_line_edit.textEdited.connect(self.on_total_time_line_edit_text_edit)

        if ('disc_updated_signal' in self.global_dict.keys() and
                        self.global_dict['disc_updated_signal'] is not None):
            self.global_dict['disc_updated_signal'].connect(self.on_disc_updated_signal)

        if ('fsp_updated_signal_1' in self.global_dict.keys() and
                        self.global_dict['fsp_updated_signal_1'] is not None):
            self.global_dict['fsp_updated_signal_1'].connect(self.on_fsp_updated_signal)

        if ('fsp_updated_signal_2' in self.global_dict.keys() and
                        self.global_dict['fsp_updated_signal_2'] is not None):
            self.global_dict['fsp_updated_signal_2'].connect(self.on_fsp_updated_signal)

    def callback_pose_stamped(self, data):
        self.last_pose_time = data.header.stamp.to_sec()
        self.last_position = dict(x=data.pose.position.x,
                                  y=data.pose.position.y,
                                  z=data.pose.position.z)
        self.last_orientation = dict(w=data.pose.orientation.w,
                                     x=data.pose.orientation.x,
                                     y=data.pose.orientation.y,
                                     z=data.pose.orientation.z)
        #t = [ data.pose.position.x, data.pose.position.y, data.pose.position.z]
        #q = [ data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z ]
        #tf_br.sendTransform(t, q,
                        #rospy.Time.now(),
                        #'px4_pose_stamped',
                        #'local_origin')
    def update_path_markers(self, qr_type="main",waypoints_moved = True):
        if self.qr_polytraj is None:
            return

        if qr_type is "main":
            qr_p = self.qr_polytraj
        elif qr_type is "entry":
            qr_p = self.qr_p_entry
        elif qr_type is "exit":
            qr_p = self.qr_p_exit

        self.update_node_path_markers()
        if self.corridors_init and waypoints_moved:
            self.corridors_init = False
            self.on_initialize_corridors_button_clicked()

        if hasattr(qr_p,'state_combined'):
            x = qr_p.state_combined['x'][0,:]
            y = qr_p.state_combined['y'][0,:]
            z = qr_p.state_combined['z'][0,:]

            if self.plot_traj_color:
                if not hasattr(self,'dist_traj'):
                    dist = self.check_distance_to_trajectory(x,y,z)
                elif len(self.dist_traj) != len(x):
                    dist = self.check_distance_to_trajectory(x,y,z)
                else:
                    dist = self.dist_traj
            else:
                dist = None

            if qr_type is "main":
                self.marker_worker.publish_path(x, y, z, dist)
            elif qr_type is "entry":
                self.marker_worker_entry.publish_path(x, y, z, dist)
            elif qr_type is "exit":
                self.marker_worker_exit.publish_path(x, y, z, dist)

            index = np.arange(0,x.size,10)
            x = qr_p.state_combined['x'][0,index]
            y = qr_p.state_combined['y'][0,index]
            z = qr_p.state_combined['z'][0,index]
            ax = qr_p.state_combined['x'][2,index]
            ay = qr_p.state_combined['y'][2,index]
            az = qr_p.state_combined['z'][2,index]

            if qr_type is "main":
                self.marker_worker.publish_quiver(x, y, z, ax, ay, az)

                # Update the entry and exit trajectories if the main loop was modified
                self.update_entry_point()
                self.update_exit_point()
            elif qr_type is "entry":
                self.marker_worker_entry.publish_quiver(x, y, z, ax, ay, az)
            elif qr_type is "exit":
                self.marker_worker_exit.publish_quiver(x, y, z, ax, ay, az)

            max_a = np.amax(np.sqrt(ax**2+ay**2+az**2))

            self.update_results_display(max_a)

            self.ppoly_laps = utils.create_complete_trajectory(self.qr_polytraj, self.qr_p_entry, self.qr_p_exit,self.entry_ID,self.exit_ID)

        else:
            print("No trajectory produced yet")


    def on_disc_updated_signal(self):
        if (self.global_dict['disc_out_waypoints'] is None or
                    'x' not in self.global_dict['disc_out_waypoints'].keys() or
                    'y' not in self.global_dict['disc_out_waypoints'].keys() or
                    'z' not in self.global_dict['disc_out_waypoints'].keys()):
            print("Spurious callback in astro; no discretized waypoints")
            return
        waypoints = self.global_dict['disc_out_waypoints']
        costs = dict()
        costs['x'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['y'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['z'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['yaw'] = [0, 0, 1]  # minimum acceleration
        # order=dict(x=9, y=9, z=9, yaw=5)
        order=dict(x=12, y=12, z=12, yaw=9)
        waypoints['yaw'] = np.array([0.0] * np.size(waypoints['x']))
        seed_times = self.seed_times#None#np.ones(waypoints['x'].size-1)*1.0
        self.qr_polytraj = traj_qr.traj_qr(waypoints,
                                            costs=costs,
                                            order=order,
                                            seed_times=seed_times,
                                            seed_avg_vel = 1.0,
                                            curv_func=self.curv_func,
                                            der_fixed=None,path_weight=None,
                                            n_samp = 500)
        # Run initial guess

        self.exit_ID = self.qr_polytraj.n_seg
        self.interactive_marker_worker.exit_ID = self.exit_ID
        # self.add_ellipsoid_constraint()
        # acceleration constraint
        # self.create_keep_in_constraint()
        self.corridors_init = False
        self.replan = False
        self.qr_polytraj.run_astro()
        self.replan = True
        self.update_path_markers()
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)
        self.obstacle_worker.make_obstacles(self.qr_polytraj.constraint_list)
        self.obstacle_worker.update_obstacles(self.qr_polytraj.constraint_list)

    def on_add_ellipsoid_constraint_click(self):
        self.add_ellipsoid_constraint()

    def add_ellipsoid_constraint(self,x0=None,A=None,rot_mat=None):

        if self.qr_polytraj is None:
            return
        constraint_type = "ellipsoid"
        params = dict()
        params['weight'] = 1e5
        params['keep_out'] = True
        params['der'] = 0
        rot_mat = np.identity(3)
        # rot_mat[0,0] = np.cos(np.pi/4)
        # rot_mat[1,1] = np.cos(np.pi/4)
        # rot_mat[0,1] = np.sin(np.pi/4)
        # rot_mat[1,0] = -np.sin(np.pi/4)
        params['rot_mat'] = rot_mat
        if x0 is None:
            params['x0'] = np.array([0,0.1,0.0])
        else:
            params['x0'] = x0

        if A is None:
            A = np.matrix(np.identity(3))
            A[0,0] = 1/0.2**2
            A[1,1] = 1/0.5**2
            A[2,2] = 1/0.3**2

            A = np.matrix(rot_mat).T*A*np.matrix(rot_mat)

        params['A'] = A

        self.qr_polytraj.add_constraint(constraint_type,params,dynamic_weighting=True)

        if not self.defer:
            self.qr_polytraj.run_astro(replan=True)
            self.update_path_markers()
            acc_wp = self.get_accel_at_waypoints("main")
            self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
            self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)
            self.obstacle_worker.make_obstacles(self.qr_polytraj.constraint_list)
            self.obstacle_worker.update_obstacles(self.qr_polytraj.constraint_list)

        # self.obstacle_worker.make_obstacles(self.qr_polytraj.constraint_list)
        # self.obstacle_worker.update_obstacles(self.qr_polytraj.constraint_list)

    def add_corridor_constraint(self,seg,r,weight=1.0,sum_func=True):
        """ Add in a corridor constraint """

        constraint_type = "cylinder"
        params = dict()
        params['x1'] = np.array([   self.qr_polytraj.waypoints['x'][0,seg],
                                    self.qr_polytraj.waypoints['y'][0,seg],
                                    self.qr_polytraj.waypoints['z'][0,seg]])
        params['x2'] = np.array([   self.qr_polytraj.waypoints['x'][0,seg+1],
                                    self.qr_polytraj.waypoints['y'][0,seg+1],
                                    self.qr_polytraj.waypoints['z'][0,seg+1]])
        params['der'] = 0
        params['l'] = r # Give the same radius buffer on the end caps
        params['r'] = r
        params['weight'] = weight
        params['keep_out'] = False
        params['active_seg'] = seg

        self.qr_polytraj.add_constraint(constraint_type,params,sum_func=sum_func,custom_weighting=False)

    def on_initialize_corridors_button_clicked(self,sig_in=None,sum_func=True):
        """ Initialise cylindrical keep in constraints around the segemnts
        """
        if self.qr_polytraj is None or self.corridors_init is True:
            return

        map = self.check_for_map()
        if map is None:
            return

        if np.size(self.qr_polytraj.constraint_list) > 0:
            self.qr_polytraj.remove_corridor_constraints()

        self.qr_polytraj.exit_on_feasible = True

        # Get minimum distance for each segment
        l_max = self.compute_nearest_distance_to_obstacles(self.qr_polytraj.waypoints)

        # find the segments currently in collision
        collision_segs = self.find_segments_in_collision()

        for i in range(self.qr_polytraj.n_seg):
            if i in collision_segs or self.all_corridors:
                # For each segment in collision
                self.add_corridor_constraint(i,l_max[i],weight=self.corridor_weight,sum_func=sum_func)


        if not self.defer:
            self.qr_polytraj.run_astro(replan=True)
            self.update_path_markers()
            acc_wp = self.get_accel_at_waypoints("main")
            # self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
            self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

        self.corridors_init = True
        self.replan = True

    def on_waypoint_control_callback(self, position, index, qr_type,defer=False):
        if qr_type is "main":
            qr_p = self.qr_polytraj
        elif qr_type is "entry":
            qr_p = self.qr_p_entry
        elif qr_type is "exit":
            qr_p = self.qr_p_exit

        if qr_p is None:
            print("not modifying waypoints; qr_polytraj is None")
            return

        if index > len(qr_p.waypoints['x'][0,:]):
            return


        if not self.controls_enabled:
            # TODO(mereweth@jpl.nasa.gov) - display error box
            # don't let user change controls
            acc_wp = self.get_accel_at_waypoints(qr_type)
            if qr_type is "main":
                self.interactive_marker_worker.update_controls(qr_p.waypoints,
                                                               indices=[index],acc_wp=acc_wp)
            elif qr_type is "entry":
                self.interactive_marker_worker_entry.update_controls(qr_p.waypoints,
                                                               indices=[index],acc_wp=acc_wp)
            elif qr_type is "exit":
                self.interactive_marker_worker_exit.update_controls(qr_p.waypoints,
                                                               indices=[index],acc_wp=acc_wp)
            return

        position["yaw"] = None

        # if qr_p.closed_loop:
        #     if index == qr_p.n_seg: # Last waypoint
        #         index = 0 # move the first waypoint instead
        #     if index == 0:
        #         # If first point being updated, also update last
        #         qr_p.update_waypoint(qr_p.n_seg,position,defer=True)


        # UPdate the waypoint
        qr_p.update_waypoint(index,position,defer=self.defer)

        acc_wp = self.get_accel_at_waypoints(qr_type)

        if qr_type is "main":
            self.interactive_marker_worker.update_controls(qr_p.waypoints,
                                                           indices=[index],closed_loop=False,acc_wp=acc_wp)
            self.qr_polytraj = qr_p
        elif qr_type is "entry":
            self.interactive_marker_worker_entry.update_controls(qr_p.waypoints,
                                                           indices=[index],closed_loop=False,acc_wp=acc_wp)
            self.qr_entry = qr_p
        elif qr_type is "exit":
            self.interactive_marker_worker_exit.update_controls(qr_p.waypoints,
                                                           indices=[index],closed_loop=False,acc_wp=acc_wp)
            self.qr_exit = qr_p

        self.update_path_markers(qr_type=qr_type)

        self.laps_set_flag = False

        self.replan = False

    def on_waypoint_menu_control_callback(self,command_type,data,qr_type):

        if command_type == "delete":
            # Delete the waypoint
            index = data['index']

            # Reset entry and exit ID
            self.entry_ID = data['entry_ID']
            self.exit_ID = data['exit_ID']

            self.delete_waypoint(index, defer=False,qr_type=qr_type)

        elif command_type == 'insert':
            # Insert the waypoint
            index = data['index']

            # Reset entry and exit ID
            self.entry_ID = data['entry_ID']
            self.exit_ID = data['exit_ID']

            # Compute the position based on the neighbouring waypoints
            new_waypoint = dict()
            for key in self.qr_polytraj.waypoints.keys():
                if index == 0:
                    # First waypoint - extend out along same vector as between the previous first two waypoints
                    new_waypoint[key] = self.qr_polytraj.waypoints[key][0,index] + (self.qr_polytraj.waypoints[key][0,index] - self.qr_polytraj.waypoints[key][0,index+1])/2
                elif index == self.qr_polytraj.n_seg + 1:
                    # New last waypoint
                    new_waypoint[key] = self.qr_polytraj.waypoints[key][0,index-1] + (self.qr_polytraj.waypoints[key][0,index-1] - self.qr_polytraj.waypoints[key][0,index-2])/2
                else:
                    new_waypoint[key] = (self.qr_polytraj.waypoints[key][0,index-1] + self.qr_polytraj.waypoints[key][0,index])/2

            new_der_fixed=dict(x=True,y=True,z=True,yaw=True)
            # TODO (bmorrell@jpl.nasa.gov) have der_Fixed as an input

            # Insert waypoint
            self.insert_waypoint(new_waypoint, index, new_der_fixed,defer=False,qr_type=qr_type)

        elif command_type == "change_entry":

            # index = data['index']

            # Reset entry ID
            self.entry_ID = data['entry_ID']

            # Update Entry
            self.update_entry_point()

        elif command_type == "change_exit":

            # index = data['index']

            # Reset entry ID
            self.exit_ID = data['exit_ID']

            # Update Entry
            self.update_exit_point()

        elif command_type == "change_both":

            # Reset entry ID
            self.entry_ID = data['entry_ID']

            # Update Entry
            self.update_entry_point()

            # Reset exit ID
            self.exit_ID = data['exit_ID']

            # Update exit
            self.update_exit_point()


        else:
            return

        # Update markers
        self.qr_polytraj.do_update_trajectory_markers = True
        self.qr_polytraj.do_update_control_yaws = True
        self.update_path_markers(qr_type=qr_type)
        acc_wp = self.get_accel_at_waypoints(qr_type)
        if qr_type is "main":
            self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
            self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)
        elif qr_type is "entry":
            self.interactive_marker_worker_entry.make_controls(self.qr_p_entry.waypoints)
            self.interactive_marker_worker_entry.update_controls(self.qr_p_entry.waypoints,acc_wp = acc_wp)
        elif qr_type is "exit":
            self.interactive_marker_worker_exit.make_controls(self.qr_p_exit.waypoints)
            self.interactive_marker_worker_exit.update_controls(self.qr_p_exit.waypoints,acc_wp = acc_wp)

        self.laps_set_flag = False

        self.replan = False

    def on_obstacle_control_callback(self, position, orientation, index,defer=False):
        if self.qr_polytraj is None:
            return

        # TODO - look at giving more controls
        self.qr_polytraj.constraint_list[index].x0[0] = position['x']
        self.qr_polytraj.constraint_list[index].x0[1] = position['y']
        self.qr_polytraj.constraint_list[index].x0[2] = position['z']

        q = np.array([orientation['w'],orientation['x'],orientation['y'],orientation['z']])
        rot_mat_old = np.matrix(self.qr_polytraj.constraint_list[index].rot_mat)
        A_old = np.matrix(self.qr_polytraj.constraint_list[index].A)
        C = rot_mat_old.T*A_old*rot_mat_old

        rot_mat = np.matrix(transforms3d.quaternions.quat2mat(q))

        A = rot_mat.T*C*rot_mat

        self.qr_polytraj.constraint_list[index].A = A
        self.qr_polytraj.constraint_list[index].rot_mat = rot_mat

        if not self.defer:
            self.qr_polytraj.run_astro(replan=True)

        self.update_path_markers()
        self.obstacle_worker.update_obstacles(self.qr_polytraj.constraint_list,
                                                       indices=[index])

    def get_state_at_time(self,t):

        if self.qr_polytraj is None:
            return

        if self.ppoly_laps is None:
            return
            # self.ppoly_laps = self.qr_polytraj.create_callable_ppoly()
            # print("ppoly computed")

        stateOut = dict()

        for key in self.ppoly_laps.keys():
            
            if key is not "yaw":
                stateOut[key] = np.zeros(5)
                stateOut[key][0] = self.ppoly_laps[key](t)
                stateOut[key][1] = self.ppoly_laps[key].derivative()(t)
                stateOut[key][2] = self.ppoly_laps[key].derivative().derivative()(t)
                stateOut[key][3] = self.ppoly_laps[key].derivative().derivative().derivative()(t)
                stateOut[key][4] = self.ppoly_laps[key].derivative().derivative().derivative().derivative()(t)
            else:
                stateOut[key] = np.zeros(3)
                stateOut[key][0] = self.ppoly_laps[key](t)
                stateOut[key][1] = self.ppoly_laps[key].derivative()(t)
                stateOut[key][2] = self.ppoly_laps[key].derivative().derivative()(t)

        return stateOut

    def on_animate_eval_callback(self, t):
        
        if self.qr_polytraj is None:
            return

        if self.ppoly_laps is None:
            return
            # self.ppoly_laps = self.qr_polytraj.create_callable_ppoly()
            # print("ppoly computed")


        t_max = self.ppoly_laps['x'].x[-1]

        if t > t_max:
            t = 0

        x = self.ppoly_laps['x'](t)
        y = self.ppoly_laps['y'](t)
        z = self.ppoly_laps['z'](t)

        # Compute yaw to be along the path
        # yaw = np.arctan2(self.ppoly_laps['y'].derivative()(t),self.ppoly_laps['x'].derivative()(t))
        # TODO Check for nans

        yaw = self.ppoly_laps['yaw'](t)

        acc_vec = np.array([self.ppoly_laps['x'].derivative().derivative()(t),
                            self.ppoly_laps['y'].derivative().derivative()(t),
                            self.ppoly_laps['z'].derivative().derivative()(t)])
        # else:
        #     t_max = utils.seg_times_to_trans_times(self.qr_polytraj.times)[-1]
        #     if t > t_max:
        #         t = 0
        #
        #     x = self.qr_polytraj.quad_traj['x'].piece_poly(t)
        #     y = self.qr_polytraj.quad_traj['y'].piece_poly(t)
        #     z = self.qr_polytraj.quad_traj['z'].piece_poly(t)
        #     yaw = self.qr_polytraj.quad_traj['yaw'].piece_poly(t)
        #
        #     acc_vec = np.array([self.qr_polytraj.quad_traj['x'].piece_poly.derivative().derivative()(t),
        #                         self.qr_polytraj.quad_traj['y'].piece_poly.derivative().derivative()(t),
        #                         self.qr_polytraj.quad_traj['z'].piece_poly.derivative().derivative()(t)])

        # q, data = body_frame.body_frame_from_yaw_and_accel( yaw, acc_vec, out_format='quaternion',deriv_type='analyse')#,x_b_current = R[:,0],y_b_current = R[:,1] )
        q, data = body_frame.body_frame_from_yaw_and_accel( yaw, acc_vec, 'quaternion' )

        # # fill message
        # msg = PoseStamped()

        # msg.header.seq = self.setpointCount
        # self.setpointCount = self.setpointCount+1
        # msg.header.stamp = rospy.get_rostime()
        # msg.header.frame_id = 1

        # msg.pose.position.x = x
        # msg.pose.position.y = y
        # msg.pose.position.z = z

        # msg.pose.orientation.w = q[0]
        # msg.pose.orientation.x = q[1]
        # msg.pose.orientation.y = q[2]
        # msg.pose.orientation.z = q[3]
        
        # # Publish message
        # self.setpoint_pub.publish(msg)
        # print("Publishing setpoint {}".format(msg))

        # print("R is {}".format(R))
        return (t_max, x, y, z, q[0], q[1], q[2], q[3])

    def insert_waypoint(self, new_waypoint, index, new_der_fixed=dict(x=True,y=True,z=True,yaw=True),defer=False, qr_type="main"):

        if qr_type is "main":
            qr_p = self.qr_polytraj
        elif qr_type is "entry":
            qr_p = self.qr_p_entry
            if index == qr_p.n_seg+2:
                print("can not insert at the end of the entry")
                return
        elif qr_type is "exit":
            qr_p = self.qr_p_exit
            if index == 0:
                print("can not insert at the start of the exit")
                return
        # Set times - halve the previous segment
        # TODO(bmorrell@jpl.nasa.gov) revisit if this is a good way to do it
        # Maybe just add more time to make the overall trajectory longer...
        if index == 0:
            # Add to the start
            times = qr_p.times[index]
            new_times = [times,times]
        elif index >= qr_p.n_seg+1:
            # Add to the end
            times = qr_p.times[qr_p.n_seg-1]
            new_times = [times,times]
        else:
            times = qr_p.times[index-1]
            new_times = [times/2.0,times/2.0]

        new_der_fixed=dict(x=True,y=True,z=True,yaw=True)
        # TODO (bmorrell@jpl.nasa.gov) have the times as an input
        print("inserting at {}\n waypoint {}\n times {}\n der_fixed, {}".format(index,new_waypoint, new_times, new_der_fixed ))
        qr_p.insert_waypoint(index,new_waypoint,new_times,new_der_fixed,defer=False)

        qr_p.do_update_trajectory_markers = True
        qr_p.do_update_control_yaws = True

        if qr_type is "main":
            self.qr_polytraj = qr_p
        elif qr_type is "entry":
            self.qr_p_entry = qr_p
        elif qr_type is "exit":
            self.qr_p_exit = qr_p

        self.laps_set_flag = False
        self.replan = False

    def delete_waypoint(self, index, defer=False, qr_type="main"):

        if qr_type is "main":
            qr_p = self.qr_polytraj
        elif qr_type is "entry":
            qr_p = self.qr_p_entry
        elif qr_type is "exit":
            qr_p = self.qr_p_exit

        if index > qr_p.n_seg + 1:
            return

        # Times. TODO: revisit how doing times. Current to just double the time for adjacent segment(s)
        if index == 0:
            # Just take the second segment time
            new_time = qr_p.times[index+1]
        elif index == qr_p.times.size:
            # Take second last segment time
            new_time = qr_p.times[index-2]
        else:
            # Add adjacent segments
            new_time = qr_p.times[index-1] + qr_p.times[index]
        # TODO (bmorrell@jpl.nasa.gov) have the times as an input.
        # Delete waypoint
        qr_p.delete_waypoint(index, new_time, defer=False)

        qr_p.do_update_trajectory_markers = True
        qr_p.do_update_control_yaws = True

        if qr_type is "main":
            self.qr_polytraj = qr_p
        elif qr_type is "entry":
            self.qr_p_entry = qr_p
        elif qr_type is "exit":
            self.qr_p_exit = qr_p

        self.laps_set_flag = False
        self.replan = False

    def on_update_times_button_clicked(self):
        """
        Update times
        """

        index = self.time_edit_index

        edit_time = self.edit_time

        self.qr_polytraj.update_times([index],edit_time,defer=self.defer)

        self.update_path_markers()
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)


    def on_scale_total_time_button_clicked(self):
        """
        Update time for the complete trajectory
        """
        if self.total_time_setting is None:
            print("Need to input time")
            return
        times = self.qr_polytraj.times.copy()

        times = times/np.sum(times)*self.total_time_setting

        self.qr_polytraj.update_times(np.arange(np.size(times)),times,defer=False)#self.defer)

        self.update_path_markers()
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)


    def on_run_astro_button_click(self):

        if self.qr_polytraj is None:
            return

        self.qr_polytraj.run_astro(replan=self.replan)

        self.replan=True

        self.update_path_markers()

        print("times are {}".format(self.qr_polytraj.times))
        print("time ratios are {}".format(self.qr_polytraj.times/np.max(self.qr_polytraj.times)))

    def on_run_astro_with_weight_growth_button_click(self):

        if self.qr_polytraj is None:
            return

        start_time = time.time()
        self.qr_polytraj.run_astro_with_increasing_weight()
        duration = time.time() - start_time

        self.replan=True

        self.update_path_markers()

    def on_optimize_button_click(self):
        if self.qr_polytraj is None:
            print("No qr_polytraj")
            return
        self.start_time = time.time()
        # copy time penalty, which is updated from input box
        self.qr_polytraj.time_penalty = self.time_penalty

        self.controls_enabled = False
        self.optimize_button.setEnabled(False)
        self.stop_optimize_button.setEnabled(True)

        self.optimizer_thread = QThread()
        self.optimizer_thread.app = self

        self.optimizer_worker = OptimizerWorker(self.on_optimizer_worker_done)
        self.optimizer_worker.local_qr_polytraj = self.qr_polytraj
        self.optimizer_worker.run_snap_opt = self.run_snap_opt
        self.optimizer_worker.moveToThread(self.optimizer_thread)
        self.optimizer_thread.started.connect(self.optimizer_worker.task)
        self.optimizer_thread.start()

    def on_mutate_optimize_button_click(self):
        if self.qr_polytraj is None:
            return

        self.qr_polytraj.mutate_optimise()

        self.replan=True

        self.update_path_markers()


    def on_stop_optimize_button_click(self):
        self.optimizer_worker.stop()
        self.optimizer_thread.quit()
        self.stop_optimize_button.setEnabled(False)

    def on_optimizer_worker_done(self):
        self.qr_polytraj = self.optimizer_worker.local_qr_polytraj
        # Reuse functionality to destroy optimizer thread
        self.on_stop_optimize_button_click()

        map = self.check_for_map()
        if map is not None:
            dist_traj = self.check_distance_to_trajectory()

            if np.min(dist_traj) < 0.0:
                print("Collisions (max {}). Planning with corridor constraints".format(np.min(dist_traj)))
                # In collision, introduce corridor constraints
                self.on_initialize_corridors_button_clicked()
                if self.defer:
                    # Run if defer stopped runnin in the corridor function above
                    self.qr_polytraj.run_astro(replan=True)
        print("planning with corridors complete")
        self.qr_polytraj.data_track.outer_opt_time = time.time() - self.start_time

        self.optimize_button.setEnabled(True)
        self.controls_enabled = True
        self.update_path_markers()
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

    def on_animate_radio_button_toggle(self, animate):
        if self.qr_polytraj is None:
            return

        if animate:
            print("start animation")
            # if self.ppoly_laps is None:
            # self.ppoly_laps = self.qr_polytraj.create_callable_ppoly()

            self.ppoly_laps = utils.create_complete_trajectory(self.qr_polytraj, self.qr_p_entry, self.qr_p_exit,self.entry_ID,self.exit_ID)
            print("ppoly computed")
            # self.ppoly_laps = utils.create_laps_trajectory(self.n_laps,
            #                                           self.qr_polytraj,
            #                                           self.qr_p_entry,
            #                                           self.qr_p_exit,
            #                                           self.entry_ID,
            #                                           self.exit_ID,
            #                                           self.qr_polytraj.closed_loop)
            self.animate_worker.start_publish()
            self.animate_worker.start_animate()
        else:
            print("stop animation")
            self.animate_worker.stop_publish()
            self.animate_worker.stop_animate()

    def update_node_path_markers(self):

        # Generate the path
        nodes = self.qr_polytraj.waypoints
        n_steps=2
        for i in range(0,nodes['x'].shape[1]-1):
            if i == 0:
                x = np.linspace(nodes['x'][0,0],nodes['x'][0,1],n_steps)
                y = np.linspace(nodes['y'][0,0],nodes['y'][0,1],n_steps)
                z = np.linspace(nodes['z'][0,0],nodes['z'][0,1],n_steps)
            else:
                x = np.concatenate([x,np.linspace(nodes['x'][0,i],nodes['x'][0,i+1],n_steps)])
                y = np.concatenate([y,np.linspace(nodes['y'][0,i],nodes['y'][0,i+1],n_steps)])
                z = np.concatenate([z,np.linspace(nodes['z'][0,i],nodes['z'][0,i+1],n_steps)])

        # get distance
        if self.plot_path_color:
            dist = self.check_distance_to_trajectory(x,y,z)
        else:
            dist = None

        self.marker_path_worker.publish_node_path(x, y, z, dist)

    def on_send_trajectory_button_click(self):
        if self.qr_polytraj is None:
            return
        msg = PolyTraj_msg()

        self.ppoly_laps = utils.create_complete_trajectory(self.qr_polytraj, self.qr_p_entry, self.qr_p_exit,self.entry_ID,self.exit_ID)

        utils.fill_n_laps_poly_traj_msg(self.ppoly_laps,msg)

        # self.qr_polytraj.fill_poly_traj_msg(msg)
        print("Sending Trajectory")
        self.polytraj_pub.publish(msg)
    def on_show_freespace_button_click(self, show_freespace):
        if self.check_for_map() is None:
            return

        if self.qr_polytraj is not None:
            # Disable Button
            if show_freespace:#self.show_freespace_radio_button.isChecked():
                # if hasattr(self,'l_max'):
                #     self.marker_path_worker.publish_tubes(self.nodes,self.l_max)
                # else:
                # TODO (bmorrell@jpl.nasa.gov) Have a check for whether or not the distances need to be updated
                self.compute_nearest_distance_to_obstacles(self.qr_polytraj.waypoints)
                self.marker_path_worker.publish_tubes(self.qr_polytraj.waypoints,self.l_max)
            else:
                # Clear markers
                self.marker_path_worker.hide_tubes(self.qr_polytraj.waypoints)
    def on_load_trajectory_button_click(self):
        filename = QFileDialog.getOpenFileName(self,
                                               'Import trajectory', '',
                                               "Traj pickle (*.traj)")
        if filename and len(filename)>0:
            filename = filename[0]
            #self.traj_file_label.setText( filename )
        else:
            print("Invalid file path")
            return
        try:
            with open(filename, 'rb') as f:
                qr_polytraj = pickle.load(f)
                if hasattr(qr_polytraj,'c_leg_poly'):
                    self.qr_polytraj = qr_polytraj
                    self.qr_polytraj.curv_func = self.curv_func
                    self.qr_polytraj.optimise_yaw_only = False
                    self.yaw_to_traj = True
                    self.qr_polytraj.set_yaw_trajectory_to_velocity_trajectory()
                    print("loaded n_seg is {}".format(self.qr_polytraj.n_seg))
                else:
                    # Convert from minsnap format
                    print("Converting loaded trajectory to astro format")
                    self.convert_from_minsnap(qr_polytraj)
        except Exception as e:
            print("Could not load pickled QRPolyTraj from {}".format(filename))
            print(e.message)
            return
        print("setting exit ID")
        self.exit_ID = self.qr_polytraj.n_seg
        self.interactive_marker_worker.exit_ID = self.exit_ID
        self.update_path_markers()
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

    def convert_from_minsnap(self,qr_polytraj):
        # Get relevant information from input polytraj
        waypoints = qr_polytraj.waypoints
        costs = qr_polytraj.costs
        order = qr_polytraj.order
        der_fixed = qr_polytraj.der_fixed
        seed_times = qr_polytraj.times
        # Initialise astro polytraj
        self.qr_polytraj = traj_qr.traj_qr(waypoints,
                                            costs=costs,
                                            order=order,
                                            seed_times=seed_times,
                                            der_fixed=der_fixed,
                                            curv_func=True,
                                            path_weight=None)
        # Run astro
        self.qr_polytraj.run_astro()

    def on_save_trajectory_button_click(self):
        filename = QFileDialog.getSaveFileName(self, 'Save trajectory', '',
                                               "Trajectory files (*.traj)")
        if filename and len(filename)>0:
            filename = filename[0]
            #self.traj_file_label.setText( filename )



        try:
            with open(filename, 'wb') as f:
                print("Saving pickled QRPolyTraj to {}".format(filename))
                qr_p_out = self.qr_polytraj
                # Remove esdf constraints
                qr_p_out.remove_esdf_constraint()
                qr_p_out.remove_nurbs_constraint()

                pickle.dump(qr_p_out, f, 2 )
        except Exception as e:
            print("Could not save pickled QRPolyTraj to {}".format(filename))
            print(e.message)

    def on_set_first_waypoint_to_drone_button_click(self):

        if self.last_orientation is None or self.last_position is None:
            print("not setting first waypoint; none received yet")
            return

        try:
            q = np.array([self.last_orientation['w'],
                 self.last_orientation['x'],
                 self.last_orientation['y'],
                 self.last_orientation['z']])

            position = self.last_position

        except AttributeError:
            print("not setting first waypoint; none received yet")
            return
        print("Updating the first and last waypoint... Position: {}\nOrientation: {}".format(position,self.last_orientation))

        # print("q is: {}".format(q))
        # yaw = tf.transformations.euler_from_quaternion(q,'rzyx')
        position['yaw'] = transforms3d.euler.quat2euler(q,'rzyx')[0]

        self.qr_polytraj.update_waypoint(0,position,defer=True)
        position['z'] += 1.0
        self.qr_polytraj.update_waypoint(self.qr_polytraj.n_seg,position,defer=self.defer)
        self.t=0 #reset visualization

        self.qr_polytraj.do_update_trajectory_markers = True
        self.qr_polytraj.do_update_control_yaws = True
        self.laps_set_flag = False

        self.update_path_markers()

        #TODO(mereweth@jpl.nasa.gov) - figure out where this is necessary
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)
        self.replan = False

    def on_set_first_waypoint_above_drone_button_click(self):

        if self.last_orientation is None or self.last_position is None:
            print("not setting first waypoint; none received yet")
            return

        try:
            q = [self.last_orientation['w'],
                 self.last_orientation['x'],
                 self.last_orientation['y'],
                 self.last_orientation['z']]

            position = self.last_position
            # Increase height by 20 cm
            position['z'] += 0.4

        except AttributeError:
            print("not setting first waypoint; none received yet")
            return
        print("Updating the first and last waypoint... Position: {}\nOrientation: {}".format(position,self.last_orientation))

        # print("q is: {}".format(q))
        # yaw = tf.transformations.euler_from_quaternion(q,'rzyx')
        position['yaw'] = transforms3d.euler.quat2euler(q,'rzyx')[0]
        position['yaw'] = self.qr_polytraj.waypoints['yaw'][0,0]

        self.qr_polytraj.update_waypoint(0,position,defer=True)
        position['z'] += 0.0
        position['yaw'] = self.qr_polytraj.waypoints['yaw'][0,-1]
        self.qr_polytraj.update_waypoint(self.qr_polytraj.n_seg,position,defer=self.defer)
        self.t=0 #reset visualization

        self.qr_polytraj.do_update_trajectory_markers = True
        self.qr_polytraj.do_update_control_yaws = True
        self.laps_set_flag = False

        self.update_path_markers()

        #TODO(mereweth@jpl.nasa.gov) - figure out where this is necessary
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

        self.replan = False

    def on_add_waypoint_button_click(self):
        if self.qr_polytraj is None:
            print("No Waypoints received yet. Need to initialise")
            return
        if self.index_waypoint > self.qr_polytraj.n_seg+1:
            # Index too large
            print("Index is too large, not adding")
            return
        # Load from GUI
        position = dict()
        position['x'] = self.new_x_waypoint
        position['y'] = self.new_y_waypoint
        position['z'] = self.new_z_waypoint
        position['yaw'] = self.new_yaw_waypoint
        index = self.index_waypoint

        self.insert_waypoint(position, index,defer=False)

        # Update Graphics
        self.update_path_markers()
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)
        self.replan = False

    def on_delete_waypoint_button_click(self):
        if self.qr_polytraj is None:
            print("No Waypoints received yet. Need to initialise")
            return
        if self.index_waypoint >= self.qr_polytraj.n_seg+1:
            # Index too large
            print("Index is too large, waypoint does not exist")
            return
        # Load from GUI
        index = self.index_waypoint

        self.delete_waypoint(index, defer=False)

        # Upd
        self.update_path_markers()
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)
        self.replan = False
    def on_x_waypoint_line_edit_text_edit(self, text):
        try:
            self.new_x_waypoint = float(text)
        except ValueError as e:
            pass
            # self.x_waypoint_line_edit.text()
            # self.x_waypoint_line_edit.setText(str(self.new_x_waypoint))
    def on_y_waypoint_line_edit_text_edit(self, text):
        try:
            self. new_y_waypoint = float(text)
        except ValueError as e:
            pass
    def on_z_waypoint_line_edit_text_edit(self, text):
        try:
            self. new_z_waypoint = float(text)
        except ValueError as e:
            pass
    def on_yaw_waypoint_line_edit_text_edit(self, text):
        try:
            self.new_yaw_waypoint = float(text)
        except ValueError as e:
            pass
    def on_collide_sphere_val_line_edit_text_edit(self, text):
        try:
            self.collide_sphere_dist = float(text)
        except ValueError as e:
            pass

    def on_accel_lim_line_edit_text_edit(self, text):
        try:
            self.accel_lim = float(text)
        except ValueError as e:
            pass
    def on_accel_weight_line_edit_text_edit(self, text):
        try:
            self.accel_weight = float(text)
        except ValueError as e:
            pass
    def on_corridor_weight_line_edit_text_edit(self, text):
        try:
            self.corridor_weight = float(text)
        except ValueError as e:
            pass
    def on_esdf_weight_line_edit_text_edit(self, text):
        try:
            self.esdf_weight = float(text)
        except ValueError as e:
            pass
    def on_index_line_edit_text_edit(self, text):
        try:
            self.index_waypoint = int(text)
        except ValueError as e:
            self.index_line_edit.text()
            # self.index_line_edit.setText(str(self.index_waypoint))
    def on_n_laps_line_edit_text_edit(self, text):
        try:
            self.n_laps = int(text)
        except ValueError as e:
            self.n_laps_line_edit.text()
            # self.index_line_edit.setText(str(self.index_waypoint))
    def on_entry_time_line_edit_text_edit(self, text):
        try:
            self.entry_time = float(text)
        except ValueError as e:
            self.entry_time_line_edit.text()
            # self.index_line_edit.setText(str(self.index_waypoint))
    def on_exit_time_line_edit_text_edit(self, text):
        try:
            self.exit_time = float(text)
        except ValueError as e:
            self.exit_time_line_edit.text()
            # self.index_line_edit.setText(str(self.index_waypoint))
    def on_time_index_line_edit_text_edit(self, text):
        try:
            self.time_edit_index = int(text)
        except ValueError as e:
            self.time_index_line_edit.text()
            # self.index_line_edit.setText(str(self.index_waypoint))
    def on_edit_time_line_edit_text_edit(self, text):
        try:
            self.edit_time = float(text)
        except ValueError as e:
            self.edit_time_line_edit.text()
    def on_total_time_line_edit_text_edit(self, text):
        try:
            self.total_time_setting = float(text)
        except ValueError as e:
            self.total_time_line_edit.text()
    def on_time_penalty_line_edit_text_edit(self, text):
        try:
            self.time_penalty = float(text)
        except ValueError as e:
            self.time_penalty_line_edit.text()
            self.time_penalty_line_edit.setText(str(self.time_penalty))
    def on_run_snap_opt_radio_button_click(self):
        if self.run_snap_opt_radio_button.isChecked():
            print("Run Snap opt is true")
            self.run_snap_opt = True
        else:
            self.run_snap_opt = False

    def on_load_obstacles_button_click(self, checked=False, filename=None):
        if self.qr_polytraj is None:
            return

        if filename is None:
            filename = QFileDialog.getOpenFileName(self,
                                                   'Import obstacles', #path,
                                                   "Obstacle YAML files (*.yaml)")
            if filename and len(filename)>0:
                filename = filename[0]
            else:
                print("Invalid file path")
                return

        # try:
        obstacle_params = utils.load_obstacles(filename)
        print("Loaded obstacles from {}".format(filename))

        # except KeyError:
        #     print("Invalid file format")
        #     return
        # except Exception as e:
        #     print("Unknown error loading obstacles from {}".format(filename))
        #     print(e)
        #     return

        for i in range(len(obstacle_params)):
            obstacle_params[i]['weight'] *= self.qr_polytraj.n_seg
            self.qr_polytraj.add_constraint(obstacle_params[i]['constraint_type'],obstacle_params[i])

        if not self.defer:
            self.qr_polytraj.run_astro(replan=True)
            self.update_path_markers()
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)
        self.obstacle_worker.make_obstacles(self.qr_polytraj.constraint_list)
        self.obstacle_worker.update_obstacles(self.qr_polytraj.constraint_list)


    def on_save_obstacles_button_click(self, checked=False, filename=None):
        if self.global_dict['disc_out_obstacles'] is None:
            print("No obstacles to save")
            return

        if filename is None:
            filename = QFileDialog.getSaveFileName(self,
                                                   'Export obstacles', #path,
                                                   "Obstacle YAML files (*.yaml)")
            if filename and len(filename)>0:
                filename = filename[0]
            else:
                print("Invalid file path")
                return

        try:
            utils.save_obstacles(self.global_dict['disc_out_obstacless'], filename)
            print("Saved waypoints to {}".format(filename))
        except KeyError:
            print("Invalid file format")
            return
        except Exception as e:
            print("Unknown error saving waypoints to {}".format(filename))
            print(e)
            return

    def load_esdf_obstacle(self,sig_in=None,sum_func=True,custom_weighting=True):
        if self.global_dict['fsp_out_map'] is None:
            print("No ESDF loaded")
            return
        print("in load_esdf_obstacle")

        if self.qr_polytraj is None:
            print("Need to have loaded an trajectory")
            return

        esdf = self.global_dict["fsp_out_map"]

        esdf_inflation_buffer = self.quad_buffer + self.inflate_buffer

        self.qr_polytraj.exit_on_feasible = True

        self.qr_polytraj.add_esdf_constraint(esdf, self.esdf_weight, self.quad_buffer,self.inflate_buffer,dynamic_weighting=False, sum_func = sum_func,custom_weighting=custom_weighting)

        if not self.defer:
            self.qr_polytraj.run_astro(replan=True)
            self.update_path_markers()
            acc_wp = self.get_accel_at_waypoints("main")
            self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
            self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

    def add_esdf_feasibility_checker(self):
        if self.global_dict['fsp_out_map'] is None:
            print("No ESDF loaded")
            return
        print("in load_esdf_obstacle")

        esdf = self.global_dict["fsp_out_map"]

        self.qr_polytraj.exit_on_feasible = True

        self.qr_polytraj.add_esdf_constraint(esdf, weight=0.0, quad_buffer=self.quad_buffer,
                                            inflate_buffer=0.0, sum_func = False,
                                            feasibility_checker=True,
                                            custom_weighting=False)


    def on_fsp_updated_signal(self):
        print('in fsp callback')
        if self.global_dict['fsp_out_map'] is None:
            print("Spurious callback in astro; no esdf loaded")
            return

        if self.qr_polytraj is not None:
            print("loading esdf obstacle")
            self.load_esdf_obstacle()

    def load_nurbs_obstacle(self,nurbs,sum_func=True,custom_weighting=True):
        
        if self.qr_polytraj is None:
            print("Need to have loaded an trajectory")
            return

        self.qr_polytraj.exit_on_feasible = True

        self.qr_polytraj.add_nurbs_constraint(nurbs, self.nurbs_weight, self.quad_buffer,self.inflate_buffer,dynamic_weighting=False, sum_func = sum_func,custom_weighting=custom_weighting)

        if not self.defer:
            self.qr_polytraj.run_astro(replan=True)
            self.update_path_markers()
            acc_wp = self.get_accel_at_waypoints("main")
            self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
            self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

    def create_two_waypoints(self):
        waypoints = dict()
        der_fixed = dict()
        waypoints['x'] = np.zeros([5,2])
        waypoints['y'] = np.zeros([5,2])
        waypoints['z'] = np.zeros([5,2])
        waypoints['yaw'] = np.zeros([3,2])

        waypoints['x'][0,:] = np.array([-1.0,1.0])
        waypoints['y'][0,:] = np.array([0.0,0.0])
        waypoints['z'][0,:] = np.array([0.0,0.0])

        der_fixed['x'] = np.zeros([5,2],dtype=bool)
        der_fixed['y'] = np.zeros([5,2],dtype=bool)
        der_fixed['z'] = np.zeros([5,2],dtype=bool)
        der_fixed['yaw'] = np.zeros([3,2],dtype=bool)

        der_fixed['x'][0,:] = True
        der_fixed['y'][0,:] = True
        der_fixed['z'][0,:] = True
        der_fixed['yaw'][0,:] = True
        costs = dict()
        costs['x'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['y'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['z'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['yaw'] = [0, 0, 1]  # minimum acceleration
        order=dict(x=9, y=9, z=9, yaw=5)
        seed_times = np.ones(waypoints['x'].shape[1]-1)*1.0
        self.qr_polytraj = traj_qr.traj_qr(waypoints,
                                            costs=costs,
                                            order=order,
                                            seed_times=seed_times,
                                            curv_func=self.curv_func,
                                            der_fixed=der_fixed,path_weight=None)

        # self.add_ellipsoid_constraint()

        # Run initial guess
        self.qr_polytraj.run_astro()
        self.update_path_markers()
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)
        self.obstacle_worker.make_obstacles(self.qr_polytraj.constraint_list)
        self.obstacle_worker.update_obstacles(self.qr_polytraj.constraint_list)


    def create_keep_in_constraint(self,der=2,limit=1e1,weight=1e5):
        """
        Create constraint on a particular derivative, spherically to a certain limit

        """
        print("Creating Keep in constraint")
        constr = dict()
        constr['constraint_type'] = "ellipsoid"
        constr['weight'] = self.accel_weight
        constr['keep_out'] = False
        constr['der'] = der
        constr['x0'] = np.zeros(3)
        A = np.matrix(np.identity(3))
        limit = self.accel_lim
        A[0,0] = 1/limit**2
        A[1,1] = 1/limit**2
        A[2,2] = 1/limit**2
        constr['rot_mat'] = np.identity(3)
        constr['A'] = A


        self.qr_polytraj.add_constraint(constr['constraint_type'],constr,dynamic_weighting=False,sum_func=True)

        # self.qr_polytraj.run_astro()
        # self.update_path_markers()
        # acc_wp = self.get_accel_at_waypoints("main")
        # self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        # self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

    def update_accel_limit(self):

        limit = self.accel_lim
        weight = self.accel_weight

        accel_lim_flag = False

        for i in range(len(self.qr_polytraj.constraint_list)):
            if self.qr_polytraj.constraint_list[i].keep_out is False and self.qr_polytraj.constraint_list[i].der == 2 and self.qr_polytraj.constraint_list[i].constraint_type is "ellipsoid":
                # Is the acceleration constraint
                # Update weight and limit
                self.qr_polytraj.constraint_list[i].weight = weight
                A = np.matrix(np.identity(3))
                limit = self.accel_lim
                A[0,0] = 1/limit**2
                A[1,1] = 1/limit**2
                A[2,2] = 1/limit**2
                self.qr_polytraj.constraint_list[i].A = A
                accel_lim_flag = True
                print("Updating accel constraint with lim {} and weight {}".format(limit, weight))

        if accel_lim_flag is False:
            # Create acceleration limit
            self.create_keep_in_constraint(der=2,limit=self.accel_lim,weight=weight)

        if not self.defer:
            self.qr_polytraj.run_astro(replan=True)
        self.update_path_markers()
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

    def update_results_display(self, max_a=-1.0):
        """
        Set text for comp time and iterations to display in the GUI
        """
        self.outer_time_line_edit.setText(str(self.qr_polytraj.data_track.outer_opt_time))
        self.comp_time_line_edit.setText(str(self.qr_polytraj.data_track.optimise_time))
        self.traj_time_line_edit.setText(str(np.sum(self.qr_polytraj.times)))
        self.traj_accel_line_edit.setText(str(max_a))
        self.iteration_line_edit.setText(str(self.qr_polytraj.data_track.iterations))
        # print("Astro cost is: {}".format(self.qr_polytraj.data_track.cost[-1]))

    def check_for_map(self):
        """ load the ESD map and checks if it exists """
        try:
            map = self.global_dict['fsp_out_map']
        except AttributeError:
            out_msg = "Not computing nearest distance. Need to have loaded an ESDF"
            print(out_msg)
            return None

        if map is None:
            out_msg = "Not computing nearest distance. Need to have loaded an ESDF"
            print(out_msg)
            return None

        return map

    def compute_nearest_distance_to_obstacles(self,waypoints):

        map = self.check_for_map()
        if map is None:
            return

        n_seg = waypoints['x'].shape[1] - 1
        n_samples = 200 # TODO (bmorrell@jpl.nasa.gov) Make this a setting in qr_polytraj
        l_max = np.zeros(n_seg)

        for k in range(0,n_seg):
            # Create query points
            query = np.matrix(np.zeros([3,n_samples]),dtype='double')
            dist = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='double')
            obs = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='int32')

            # load in x, y, z points
            query[0,:] = np.around(np.linspace(waypoints['x'][0,k],waypoints['x'][0,k+1],n_samples),4)
            query[1,:] = np.around(np.linspace(waypoints['y'][0,k],waypoints['y'][0,k+1],n_samples),4)
            query[2,:] = np.around(np.linspace(waypoints['z'][0,k],waypoints['z'][0,k+1],n_samples),4)

            # Query the database
            map.getDistanceAtPosition(query, dist, obs)

            if self.unknown_is_free:
                dist[obs != 1] = 2.0

            # Find the minimum obstacle distance
            l_max[k] = np.min(dist) - self.quad_buffer

        # check for collisions
        if np.min(l_max) <=0.0:
            out_msg = "Path is in collision with environment"
            print(out_msg)
            # self.terminal_text_edit.appendPlainText(out_msg)

        # Update l_max in the qr_polytraj object
        # print("setting l_max")
        self.l_max = l_max
        return l_max

    def check_distance_to_trajectory(self,x=None,y=None,z=None,samp_mult=100):
        map = self.check_for_map()
        if map is None:
            return

        if x is None:


            if not hasattr(self.qr_polytraj,'state_combined'):
                self.qr_polytraj.get_trajectory()

            x = self.qr_polytraj.state_combined['x'][0,:]
            y = self.qr_polytraj.state_combined['y'][0,:]
            z = self.qr_polytraj.state_combined['z'][0,:]

        # Create query points
        query = np.matrix(np.zeros([3,x.size]),dtype='double')
        dist = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='double')
        dist2 = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='double')
        obs = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='int32')
        obs2 = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='int32')

        # load in x, y, z points
        query[0,:] = np.around(x,4)
        query[1,:] = np.around(y,4)
        query[2,:] = np.around(z,4)

        # Query the database
        # map.getDistanceAtPosition(query, dist2, obs2)
        grad = np.matrix(np.zeros(np.shape(query)),dtype='double')
        map.getDistanceAndGradientAtPosition(query, dist, grad, obs)

        # print("dist 1 is {}\ndist 2 is {}\nobs 1 is {}\n obs 2 is {}\ngrad is {}".format(np.min(dist),np.min(dist2),np.sum(obs),np.sum(obs2),grad))
        # Add buffer on quad:
        dist -= self.quad_buffer

        if self.unknown_is_free:
            dist[obs != 1] = 2.0

        # return the distance
        return dist

    def on_check_distance_button_click(self):
        """ Checks for the minimum distance from the trajectory to obstacles and
        prints the value in a text field
        """

        map = self.check_for_map()
        if map is None:
          return

        if self.qr_polytraj is not None:
          # if self.distance_current
          # Get distance to the trajectory
          dist_traj = self.check_distance_to_trajectory()
          # self.dist_traj = dist_traj
          print("Minimum Distance from Trajectory to obstacles is: {} m".format(np.min(dist_traj)))
          self.traj_dist_line_edit.setText("{:.4f}".format(np.min(dist_traj)))
        else:
          print('No trajectory to Check')
          self.traj_dist_line_edit.setText("N/A")
          return

    def on_show_collision_checkbox_toggled(self, show_collision):

        if self.qr_polytraj is None:
            return

        map = self.check_for_map()
        if map is None:
          return

        grad_tol = 1e-14
        dist_tol = self.collide_sphere_dist

        # Toggled on:
        if show_collision:#self.show_freespace_radio_button.isChecked():
            # Generate x, y, z points
            if not hasattr(self.qr_polytraj,'state_combined'):
                self.qr_polytraj.get_trajectory()

            x = self.qr_polytraj.state_combined['x'][0,:]
            y = self.qr_polytraj.state_combined['y'][0,:]
            z = self.qr_polytraj.state_combined['z'][0,:]

            # Compute distance to trajectory
            dist_traj = np.array(self.check_distance_to_trajectory(x, y, z).flat)
            # Gradient of distance along the trajectory
            grad_traj = np.gradient(dist_traj)
            curv_traj = np.gradient(grad_traj)

            # print("X is {}".format(x))
            # print(dist_traj)
            # print(grad_traj)
            # print(np.min(np.abs(grad_traj)))

            # Turning points with distance below threshold
            # - where gradient near zero and curvature positive (minima)
            # tp = np.where(((np.abs(grad_traj)<grad_tol)*(dist_traj<dist_tol)*(curv_traj>0.0)))


            tp = np.where(dist_traj<dist_tol)

            points = dict()
            points['x'] = x[tp]
            points['y'] = y[tp]
            points['z'] = z[tp]


            self.marker_worker.publish_marker_spheres(points)
        else:
            # Clear markers
            self.marker_worker.hide_marker_spheres()
            print("Turning off")

    def find_segments_in_collision(self,samp_mult=100):

        collision = np.zeros(self.qr_polytraj.n_seg,dtype=bool)

        for i in range(self.qr_polytraj.n_seg):
            # Generate x, y, z points

            if hasattr(self.qr_polytraj,'state'):
                x = self.qr_polytraj.state['x'][0,:,i]
                y = self.qr_polytraj.state['y'][0,:,i]
                z = self.qr_polytraj.state['z'][0,:,i]

            dist = self.check_distance_to_trajectory(x,y,z)

            if np.min(dist)<0.0:
                collision[i] = True

        return np.where(collision)[0]

    def on_add_takeoff_button_click(self):
        self.add_take_off()

    def on_add_landing_button_click(self):
        self.add_landing()

    def add_take_off(self, entry_waypoints=None, entry_times=None):
        """ Adding a take-off trajectory """

        entry_ID = self.entry_ID

        entry_time_unit = self.entry_time

        if entry_waypoints is None:
          # Default load
          if self.last_position is not None:
              # Set start to above drone
              entry_waypoints = dict()
              for key in self.last_position.keys():
                  entry_waypoints[key] = self.last_position[key]

              entry_waypoints['z'] += 0.05

              q = np.array([self.last_orientation['w'],
                   self.last_orientation['x'],
                   self.last_orientation['y'],
                   self.last_orientation['z']])

              entry_waypoints['yaw'] = transforms3d.euler.quat2euler(q,'rzyx')[0]

              # TODO (bmorrell@jpl.nasa.gov) make this an input
              entry_times = np.array([entry_time_unit])
          else:
              # set to a default 0 0 0
              entry_waypoints = dict(x=0.0,y=0.0,z=0.0,yaw=0.0)
              entry_waypoints['z'] += 0.2

          # TODO (bmorrell@jpl.nasa.gov) make this an input
          entry_times = np.array([entry_time_unit])
        elif entry_times is None:
          entry_times = np.array([entry_time_unit]*entry_waypoints['x'].shape[1])

        #Set last waypoints
        entry_waypoints = utils.form_waypoints_polytraj(entry_waypoints,self.qr_polytraj.n_der)
        entry_waypoints, entry_der_fixed = utils.form_entry_or_exit_waypoints("entry",self.qr_polytraj, entry_waypoints, entry_ID, add_on_der_fixed=dict(x=None,y=None,z=None,yaw=None))
        print("entry waypoints are: {}".format(entry_waypoints))
        # Create new qr_polytraj
        self.qr_p_entry = traj_qr.traj_qr(entry_waypoints,
                                            costs=self.qr_polytraj.costs,order=self.qr_polytraj.order,
                                            seed_times=entry_times,
                                            curv_func=self.curv_func,
                                            der_fixed=entry_der_fixed,path_weight=None,yaw_to_traj=False)
        # self.qr_p_entry.initial_guess()
        self.qr_p_entry.run_astro()

        # Create Controls and trajectory for display
        self.update_path_markers(qr_type="entry")
        self.interactive_marker_worker_entry.make_controls(self.qr_p_entry.waypoints)
        acc_wp = self.get_accel_at_waypoints("entry")
        self.interactive_marker_worker_entry.update_controls(self.qr_p_entry.waypoints,acc_wp = acc_wp)

    def add_landing(self, exit_waypoints=None, exit_times=None, add_on_der_fixed=dict(x=None,y=None,z=None,yaw=None)):
        """ Adding a landing trajectory """

        exit_ID = self.exit_ID

        exit_time_unit = self.exit_time

        if exit_waypoints is None:
            # Default load
            if self.last_position is not None:
                # Set start to above drone
                exit_waypoints = dict()
                for key in self.last_position.keys():
                    exit_waypoints[key] = self.last_position[key]

                exit_waypoints['z'] += 0.4

                q = np.array([self.last_orientation['w'],
                     self.last_orientation['x'],
                     self.last_orientation['y'],
                     self.last_orientation['z']])

                exit_waypoints['yaw'] = transforms3d.euler.quat2euler(q,'rzyx')[0]

                # TODO (bmorrell@jpl.nasa.gov) make this an input
                exit_times = np.array([exit_time_unit])
            else:
                # set to a default 0 0 0
                exit_waypoints = dict(x=0.0,y=0.0,z=0.0,yaw=0.0)

            # TODO (bmorrell@jpl.nasa.gov) make this an input
            exit_times = np.array([exit_time_unit])
        elif exit_times is None:
            exit_times = np.array([exit_time_unit]*exit_waypoints['x'].shape[1])

        #Set last waypoints
        exit_waypoints = utils.form_waypoints_polytraj(exit_waypoints,self.qr_polytraj.n_der)
        exit_waypoints, exit_der_fixed = utils.form_entry_or_exit_waypoints("exit",self.qr_polytraj, exit_waypoints, exit_ID, add_on_der_fixed=add_on_der_fixed)
        print("exit waypoints are: {}".format(exit_waypoints))
        # Create new qr_polytraj
        self.qr_p_exit = traj_qr.traj_qr(exit_waypoints,
                                            costs=self.qr_polytraj.costs,order=self.qr_polytraj.order,
                                            seed_times=exit_times,
                                            curv_func=self.curv_func,
                                            der_fixed=exit_der_fixed,path_weight=None,yaw_to_traj=False)
        self.qr_p_exit.run_astro()

        # Create Controls and trajectory for display
        print("forming landing")
        self.update_path_markers(qr_type="exit")
        self.interactive_marker_worker_exit.make_controls(self.qr_p_exit.waypoints)
        acc_wp = self.get_accel_at_waypoints("exit")
        self.interactive_marker_worker_exit.update_controls(self.qr_p_exit.waypoints,acc_wp=acc_wp)

    def update_entry_point(self):
        if self.qr_p_entry is None:
            return

        entry_ID = self.entry_ID

        # Get new state at the waypoint
        new_waypoint = utils.get_state_at_waypoint(self.qr_polytraj, entry_ID)
        index = self.qr_p_entry.waypoints['x'].shape[1]-1

        # Update trajectory
        self.qr_p_entry.update_waypoint(index,new_waypoint,defer=False)

        self.update_path_markers(qr_type='entry')

    def update_exit_point(self):
        if self.qr_p_exit is None:
            return

        exit_ID = self.exit_ID

        # Get new state at the waypoint
        new_waypoint = utils.get_state_at_waypoint(self.qr_polytraj, exit_ID)
        index = 0

        # Update trajectory
        self.qr_p_exit.update_waypoint(index,new_waypoint,defer=False)

        self.update_path_markers(qr_type='exit')

    def on_set_takeoff_and_landing_to_drone_click(self):

        # Get current position
        if self.last_position is not None:
            # Set start to above drone
            print(self.last_position)
            position = dict()
            for key in self.last_position.keys():
                position[key] = self.last_position[key]

            q = np.array([self.last_orientation['w'],
                 self.last_orientation['x'],
                 self.last_orientation['y'],
                 self.last_orientation['z']])

            position['yaw'] = transforms3d.euler.quat2euler(q,'rzyx')[0]

        else:
            # set to a default 0 0 0
            position = dict(x=0.0,y=0.0,z=0.0,yaw=0.0)

        # Take-off
        self.add_take_off()

        # landing
        self.set_landing_sequence(position)

        # Take-off - re-run
        self.add_take_off()

    def set_landing_sequence(self, position):
        # Assumes position is where the drone is or where you want to landing

        exit_waypoints = dict()
        add_on_der_fixed = dict()

        # number of waypoints
        n_way = 3

        print(position)

        for key in position.keys():

            # initilise
            if key is 'z':
                exit_waypoints[key] = np.array([position[key]+0.4,position[key]+0.2,position[key]-3.0])
            else:
                exit_waypoints[key] = np.array([position[key]]*3)
                exit_waypoints[key][0:2] += np.random.randn(1)*1e-5

            n_der = self.qr_polytraj.der_fixed[key].shape[0]
            add_on_der_fixed[key] = np.ones((n_der,3),dtype=bool)
            # add_on_der_fixed[key][:-1,1:] = False


        # Set for a long final time to descend slowly
        # exit_times = np.array([2.0,1.0,10.0])
        exit_times = np.array([self.exit_time,1.0,10.0])

        print(exit_waypoints['x'].shape)

        self.add_landing(exit_waypoints, exit_times, add_on_der_fixed)

    def get_accel_at_waypoints(self,qr_type="main"):
        if self.qr_polytraj is None:
            return

        if qr_type is "main":
            qr_p = self.qr_polytraj
            if not hasattr(qr_p,'state'):
                qr_p.get_trajectory()

            # Get the state at the start of each segment and the end of the last segment
            ax = np.append(qr_p.state['x'][2,0,:],qr_p.state['x'][2,-1,-1])
            ay = np.append(qr_p.state['y'][2,0,:],qr_p.state['x'][2,-1,-1])
            az = np.append(qr_p.state['z'][2,0,:],qr_p.state['x'][2,-1,-1])

            return np.reshape(np.array([ax,ay,az]),(3,np.size(ax)))
        elif qr_type is "entry":
            # qr_p = self.qr_p_entry
            return None
        elif qr_type is "exit":
            # qr_p = self.qr_p_exit
            return None

    def on_save_waypoints_from_traj_click(self, checked=False, filename=None):
        if self.qr_polytraj is None:
            print("No traj to save")
            return

        if filename is None:
            filename = QFileDialog.getSaveFileName(self,
                                                   'Export waypoints', #path,
                                                   "Waypoint YAML files (*.yaml)")
            if filename and len(filename)>0:
                filename = filename[0]
            else:
                print("Invalid file path")
                return

        try:
            out_waypoints = dict()
            for key in self.qr_polytraj.waypoints.keys():
                out_waypoints[key] = self.qr_polytraj.waypoints[key][0,:]
            utils.save_waypoints(out_waypoints, filename)
            print("Saved waypoints to {}".format(filename))
        except KeyError:
            print("Invalid file format")
            return
        except Exception as e:
            print("Unknown error saving waypoints to {}".format(filename))
            print(e)
            return

    def on_set_yaw_to_traj_button_click(self):

        self.qr_polytraj.set_yaw_trajectory_to_velocity_trajectory()

        # if not self.defer:
            # self.qr_polytraj.run_astro()
        self.update_path_markers()
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

    def on_set_yaw_to_zero_button_click(self):
        self.qr_polytraj.set_yaw_to_zero()

        if not self.defer:
            self.qr_polytraj.run_astro()
            self.update_path_markers()
            acc_wp = self.get_accel_at_waypoints("main")
            self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
            self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

    def on_corridor_weight_update_button_clicked(self):

        self.qr_polytraj.set_constraint_weight(self.corridor_weight,"cylinder")

    def on_esdf_weight_update_button_clicked(self):

        self.qr_polytraj.set_constraint_weight(self.esdf_weight,"esdf")

    def on_nurbs_weight_update_button_clicked(self):

        self.qr_polytraj.set_constraint_weight(self.nurbs_weight,"nurbs")




def main():
    from torq_gcs.plan.astro_plan import QRPolyTrajGUI
    app = QApplication( sys.argv )
    global_dict = dict()
    astro = QRPolyTrajGUI(global_dict, new_node=True)
    astro.show()
    return app.exec_()
if __name__=="__main__":
    sys.exit(main())
