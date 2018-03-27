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

import numpy
np = numpy
import scipy
from scipy import interpolate
sp = scipy
import time

from minsnap import utils, quadrotor_polytraj
from diffeo import body_frame
from diffeo import angular_rates_accel
from diffeo import controls

import rospkg, rospy, tf
from python_qt_binding import loadUi
from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *
from python_qt_binding.QtWidgets import *

#from torq_gcs import plan
from torq_gcs.plan.ros_helpers import TrajectoryDisplayWorker
from torq_gcs.plan.ros_helpers import WaypointControlWorker
from torq_gcs.plan.ros_helpers import AnimateWorker

from px4_msgs.msg import PolyTraj as PolyTraj_msg
from geometry_msgs.msg import PoseStamped

import transforms3d
from transforms3d import euler

class OptimizerWorker(QObject):
    finished = Signal() # class variable shared by all instances

    def __init__(self, finished_callback, parent=None):
        super(OptimizerWorker, self).__init__(parent)
        self.finished.connect(finished_callback)
        self.is_running = True

    def task(self):
        print("Optimizing outer loop with time {}".format(self.local_qr_polytraj.time_penalty))
        old_times = self.local_qr_polytraj.times
        converged = False

        while not converged and self.is_running:
            res, modified = self.local_qr_polytraj.relative_time_opt(
                                                     method='COBYLA',
                                                     options=dict(disp=3,
                                                                  maxiter=100,
                                                                  tol=0.1))
            #qr_p.relative_time_opt(method='BFGS', options=dict(disp=3, maxiter=1))
            #qr_p.relative_time_opt(method='Nelder-Mead', options=dict(disp=3, maxiter=1, tol=0.5))
            #qr_p.relative_time_opt(method='SLSQP', options=dict(disp=3, maxiter=1, tol=0.5))

            converged = res.success
        self.finished.emit()

    def stop(self):

        self.is_running = False

class QRPolyTrajGUI(QWidget):
    def __init__(self,
                 global_dict,
                 node_name='unco',
                 new_node=False,
                 parent=None):
        super(QRPolyTrajGUI, self).__init__(parent)

        ui_file = os.path.join(rospkg.RosPack().get_path('torq_gcs'),
                               'resource', 'plan', 'Minsnap.ui')
        loadUi(ui_file, self)

        self.qr_polytraj = None
        self.qr_p_entry = None
        self.qr_p_exit = None
        self.time_penalty = 1000
        self.ppoly_laps = None
        self.set_yaw = False

        # Settings for showing distance to obstacles
        self.unknown_is_free = True
        self.quad_buffer = 0.6
        self.plot_traj_color = True

        self.entry_ID = 0
        self.exit_ID = 0
        self.n_laps = 1
        self.entry_time = 5
        self.exit_time = 5
        self.laps_set_flag = False # indicates whether the laps/entry and exit have been set

        self.time_edit_index = 0
        self.edit_time = 1.0
        self.total_time_setting = None

        self.last_pose_time   = None
        self.last_position    = None
        self.last_orientation = None

        self.controls_enabled = True

        self.new_x_waypoint = 0.
        self.new_y_waypoint = 0.
        self.new_z_waypoint = 0.
        self.new_yaw_waypoint = 0.
        self.index_waypoint = 0
        self.collide_sphere_dist = 0.0

        self.yaw_rand_delta = 0.0
        self.yaw_constant = 0.0

        self.quad_params = None

        self.a_max = 0.0
        self.rpm_max = 0.0
        self.M_max = 0.0
        self.v_max = 0.0
        self.thr_max = 0.0

        self.rpm_max_all = 0
        self.rpm_min_all = 90000
        self.vel_max_all = 0
        self.acc_max_all = 0
        self.thrust_max_all = 0

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

        self.interactive_marker_worker = WaypointControlWorker(
                                            self.on_waypoint_control_callback,
                                            menu_callback=self.on_waypoint_menu_control_callback,
                                            frame_id="local_origin",
                                            marker_topic="trajectory_control",
                                            qr_type="main")
        self.interactive_marker_worker.moveToThread(self.ros_helper_thread)

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

        self.polytraj_pub = rospy.Publisher('poly_traj_input', PolyTraj_msg,
                                            queue_size=1)
        self.pose_sub = rospy.Subscriber("pose_stamped_out", PoseStamped,
                                         self.callback_pose_stamped)


        # TODO(mereweth@jpl.nasa.gov) - what data products come out of plan?
        #    global_dict = dict(plan_out_SOMETHING=None)
        #else:
            # warn about overwriting?
            # if 'plan_out_SOMETHING' in global_dict.keys():
        #    global_dict['plan_out_SOMETHING'] = None
        self.global_dict = global_dict

        self.animate_checkbox.toggled.connect(self.on_animate_radio_button_toggle)
        self.close_loop_checkbox.toggled.connect(self.on_close_loop_checkbox_toggle)
        self.send_trajectory_button.clicked.connect(self.on_send_trajectory_button_click)
        self.load_trajectory_button.clicked.connect(self.on_load_trajectory_button_click)
        self.save_trajectory_button.clicked.connect(self.on_save_trajectory_button_click)
        self.optimize_button.clicked.connect(self.on_optimize_button_click)
        self.stop_optimize_button.clicked.connect(self.on_stop_optimize_button_click)
        self.set_first_waypoint_to_drone_button.clicked.connect(self.on_set_first_waypoint_to_drone_button_click)
        self.set_first_waypoint_above_drone_button.clicked.connect(self.on_set_first_waypoint_above_drone_button_click)

        self.save_waypoints_from_traj_button.clicked.connect(self.on_save_waypoints_from_traj_click)
        self.on_load_quad_params_button.clicked.connect(self.on_load_quad_parameters)

        self.set_start_and_end.clicked.connect(self.on_set_takeoff_and_landing_to_drone_click)
        self.add_takeoff_button.clicked.connect(self.on_add_takeoff_button_click)
        self.add_landing_button.clicked.connect(self.on_add_landing_button_click)

        self.create_n_laps_button.clicked.connect(self.on_create_n_laps_button_click)

        self.on_set_yaw_to_traj_button.clicked.connect(self.set_yaw_to_traj_vel)
        self.set_yaw_to_rand_button.clicked.connect(self.on_set_yaw_to_random_button_click)
        self.set_constant_yaw_button.clicked.connect(self.on_set_yaw_to_constant_button_click)

        self.plan_with_obstacles_button.clicked.connect(self.on_plan_with_obstacles_button_click)

        self.add_waypoint_button.clicked.connect(self.on_add_waypoint_button_click)
        self.delete_waypoint_button.clicked.connect(self.on_delete_waypoint_button_click)

        self.check_distance_button.clicked.connect(self.on_check_distance_button_click)
        self.show_collision_checkbox.toggled.connect(self.on_show_collision_checkbox_toggled)

        self.change_segment_time.clicked.connect(self.on_update_times_button_clicked)
        self.change_total_time.clicked.connect(self.on_scale_total_time_button_clicked)

        double_validator_time = QDoubleValidator(parent=self.time_penalty_line_edit)
        double_validator_time.setBottom(0)
        self.time_penalty_line_edit.setValidator(double_validator_time)
        self.time_penalty_line_edit.setText(str(self.time_penalty))
        self.time_penalty_line_edit.textEdited.connect(self.on_time_penalty_line_edit_text_edit)

        self.traj_dist_line_edit.setText("N/A")

        self.comp_time_line_edit.setValidator(double_validator_time)

        self.outer_time_line_edit.setValidator(double_validator_time)

        self.traj_time_line_edit.setValidator(double_validator_time)
        self.traj_accel_line_edit.setValidator(double_validator_time)

        double_validator_wayp = QDoubleValidator(parent=self.x_waypoint_line_edit)
        # rx = QRegExp("[+-]?[1-9]\\d+\\.\\d{2}")
        # double_validator_wayp = QRegExpValidator(rx,parent=self.y_waypoint_line_edit)
        # double_validator_wayp.setRange(-10.0, 100.0, 5)
        double_validator_wayp.setBottom(-100.0)
        double_validator_wayp.setTop(100.0)
        self.x_waypoint_line_edit.setValidator(double_validator_wayp)
        self.x_waypoint_line_edit.setText("0")
        self.x_waypoint_line_edit.textEdited.connect(self.on_x_waypoint_line_edit_text_edit)
        # double_validator_y = QDoubleValidator(parent=self.y_waypoint_line_edit)
        self.y_waypoint_line_edit.setValidator(double_validator_wayp)
        self.y_waypoint_line_edit.setText("0")
        self.y_waypoint_line_edit.textEdited.connect(self.on_y_waypoint_line_edit_text_edit)
        # double_validator_z = QDoubleValidator(parent=self.z_waypoint_line_edit)
        self.z_waypoint_line_edit.setValidator(double_validator_wayp)
        self.z_waypoint_line_edit.setText("0")
        self.z_waypoint_line_edit.textEdited.connect(self.on_z_waypoint_line_edit_text_edit)
        # double_validator_yaw = QDoubleValidator(parent=self.yaw_waypoint_line_edit)
        self.yaw_waypoint_line_edit.setValidator(double_validator_wayp)
        self.yaw_waypoint_line_edit.setText("0")
        self.yaw_waypoint_line_edit.textEdited.connect(self.on_yaw_waypoint_line_edit_text_edit)

        self.collide_sphere_val_line_edit.setValidator(double_validator_wayp)
        self.collide_sphere_val_line_edit.setText("0")
        self.collide_sphere_val_line_edit.textEdited.connect(self.on_collide_sphere_val_line_edit_text_edit)

        int_validator_idx = QIntValidator(0,200,parent=self.index_line_edit)
        self.index_line_edit.setValidator(int_validator_idx)
        self.index_line_edit.setText("0")
        self.index_line_edit.textEdited.connect(self.on_index_line_edit_text_edit)

        # int_validator_n_laps = QIntValidator(0,200,parent=self.n_laps_line_edit)
        self.n_laps_line_edit.setValidator(int_validator_idx)
        self.n_laps_line_edit.setText(str(self.n_laps))
        self.n_laps_line_edit.textEdited.connect(self.on_n_laps_line_edit_text_edit)

        # double_validator_entry_time = QDoubleValidator(parent=self.entry_time_line_edit)
        self.entry_time_line_edit.setValidator(double_validator_wayp)
        self.entry_time_line_edit.setText(str(self.entry_time))
        self.entry_time_line_edit.textEdited.connect(self.on_entry_time_line_edit_text_edit)

        # double_validator_exit_time = QDoubleValidator(parent=self.exit_time_line_edit)
        self.exit_time_line_edit.setValidator(double_validator_wayp)
        self.exit_time_line_edit.setText(str(self.exit_time))
        self.exit_time_line_edit.textEdited.connect(self.on_exit_time_line_edit_text_edit)

        self.time_index_line_edit.setValidator(int_validator_idx)
        self.time_index_line_edit.setText("0")
        self.time_index_line_edit.textEdited.connect(self.on_time_index_line_edit_text_edit)

        self.edit_time_line_edit.setValidator(double_validator_wayp)
        self.edit_time_line_edit.setText(str(self.edit_time))
        self.edit_time_line_edit.textEdited.connect(self.on_edit_time_line_edit_text_edit)

        self.yaw_delta_line_edit.setValidator(double_validator_wayp)
        self.yaw_delta_line_edit.setText(str(self.yaw_rand_delta))
        self.yaw_delta_line_edit.textEdited.connect(self.on_yaw_rand_line_edit_text_edit)

        self.const_yaw_line_edit.setValidator(double_validator_wayp)
        self.const_yaw_line_edit.setText(str(self.yaw_constant))
        self.const_yaw_line_edit.textEdited.connect(self.on_const_yaw_line_edit_text_edit)

        self.total_time_line_edit.setValidator(double_validator_wayp)
        # self.edit_time_line_edit.setText(str(self.edit_time))
        self.total_time_line_edit.textEdited.connect(self.on_total_time_line_edit_text_edit)

        if ('disc_updated_signal' in self.global_dict.keys() and
                        self.global_dict['disc_updated_signal'] is not None):
            self.global_dict['disc_updated_signal'].connect(self.on_disc_updated_signal)
            #print("Connecting disc_updated_signal")
        # Trajectory Validator
        # Create a table
        self.table = QTableWidget(parent)
        self.table.setRowCount(1)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Min Dist2Obs", "Max RPM", "Min RPM", "Max Vel", "Max Acc", "Max Thrust"])
        # Add buttons to last row
        self.obs_btn = QPushButton(self.table)
        self.obs_btn.setText('(Re)-calculate')
        self.act_btn = QPushButton(self.table)
        self.act_btn.setText('(Re)-calculate')
        self.obs_btn.clicked.connect(self.update_dist2obs)
        self.act_btn.clicked.connect(self.update_act)
        self.table.setCellWidget(0, 0, self.obs_btn)
        self.table.setCellWidget(0, 1, self.act_btn)
        self.table.resizeColumnsToContents()
        # Add table to layout
        self.horizontalLayout_trajeditor.addWidget(self.table)

    def update_dist2obs(self):
        if self.qr_polytraj is None:
            print("Need trajectory")
            return
        if self.check_for_map() is None:
            print("Need ESDF Loaded")
            return
        safe = True
        self.find_segments_in_collision()
        self.table.setRowCount(len(self.min_dist)+1)
        for i in range(len(self.min_dist)):
            self.table.setItem(1+i, 0, QTableWidgetItem('%.2f' % self.min_dist[i]))
            if self.min_dist[i] < 0:
                self.table.item(1+i, 0).setBackground(QColor(255,0,0))
                safe = False
            else:
                self.table.item(1+i, 0).setBackground(QColor(0,255,0))

        self.table.setVerticalHeaderLabels(map(str,range(self.table.rowCount())))
        if safe == True:
            self.obs_btn.setStyleSheet("background-color: green; color: white;")
        else:
            self.obs_btn.setStyleSheet("background-color: red; color: white;")

    def update_act(self):
        self.on_check_maximum_performance()
        safe = True
        if self.qr_polytraj is None:
            print("Need polytraj loaded")
            return
        if self.quad_params is None:
            print("Need to load quad params")
            return
        # Assumes all act params have same number of elements
        row_num = max(len(self.rpm_max_all),len(self.vel_max_all),len(self.acc_max_all),len(self.thrust_max_all)) + 1
        self.table.setRowCount(row_num)

        params = [self.rpm_max_all, self.rpm_min_all, self.vel_max_all, self.acc_max_all, self.thrust_max_all]
        param_bounds = [10000, 100, 1.5, 3, 10]
        for j in range(len(params)):
            for i in range(len(params[j])):
                self.table.setItem(1+i, 1+j, QTableWidgetItem('%.2f' % params[j][i]))
                # lower bounds for j==1 else upper bound
                if ((params[j][i] < param_bounds[j]) if j==1 else (params[j][i]>param_bounds[j])):
                    self.table.item(1+i, 1+j).setBackground(QColor(255,0,0))
                    safe = False
                else:
                    self.table.item(1+i, 1+j).setBackground(QColor(0,255,0))
            self.table.setVerticalHeaderLabels(map(str,range(self.table.rowCount())))
            if safe == True:
                self.act_btn.setStyleSheet("background-color: green; color: white;")
            else:
                self.act_btn.setStyleSheet("background-color: red; color: white;")


    def on_time_penalty_line_edit_text_edit(self, text):
        try:
            self.time_penalty = float(text)
        except ValueError as e:
            self.time_penalty_line_edit.text()
            self.time_penalty_line_edit.setText(str(self.time_penalty))

    def on_x_waypoint_line_edit_text_edit(self, text):
        try:
            self.new_x_waypoint = float(text)
        except ValueError as e:
            pass
            # self.x_waypoint_line_edit.text()
            # self.x_waypoint_line_edit.setText(str(self.new_x_waypoint))
    def on_y_waypoint_line_edit_text_edit(self, text):
        try:
            self.new_y_waypoint = float(text)
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

    def on_index_line_edit_text_edit(self, text):
        try:
            self.index_waypoint = int(text)
        except ValueError as e:
            self.index_line_edit.text()
            # self.index_line_edit.setText(str(self.index_waypoint))
    def on_n_laps_line_edit_text_edit(self, text):
        try:
            self.n_laps = int(text)
            if self.n_laps<1:
                self.n_laps = 1
                print("number of laps set to 1")
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
            # self.index_line_edit.setText(str(self.index_waypoint))
    def on_total_time_line_edit_text_edit(self, text):
        try:
            self.total_time_setting = float(text)
        except ValueError as e:
            self.total_time_line_edit.text()
    def on_yaw_rand_line_edit_text_edit(self, text):
        try:
            self.yaw_rand_delta = float(text)
        except ValueError as e:
            self.yaw_delta_line_edit.text()
            # self.index_line_edit.setText(str(self.index_waypoint))
    def on_const_yaw_line_edit_text_edit(self, text):
        try:
            self.yaw_constant = float(text)
        except ValueError as e:
            self.const_yaw_line_edit.text()
            # self.index_line_edit.setText(str(self.index_waypoint))
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

    def update_path_markers(self,qr_type="main"):
        if self.qr_polytraj is None:
            return

        if qr_type is "main":
            qr_p = self.qr_polytraj
        elif qr_type is "entry":
            qr_p = self.qr_p_entry
        elif qr_type is "exit":
            qr_p = self.qr_p_exit

        trans_times = utils.seg_times_to_trans_times(qr_p.times)
        t_total = trans_times[-1] - trans_times[0]
        t = np.linspace(trans_times[0], trans_times[-1], t_total*100)
        x = qr_p.quad_traj['x'].piece_poly(t)
        y = qr_p.quad_traj['y'].piece_poly(t)
        z = qr_p.quad_traj['z'].piece_poly(t)

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

        t = np.linspace(trans_times[0], trans_times[-1], t_total*30)
        x = qr_p.quad_traj['x'].piece_poly(t)
        y = qr_p.quad_traj['y'].piece_poly(t)
        z = qr_p.quad_traj['z'].piece_poly(t)
        ax = qr_p.quad_traj['x'].piece_poly.derivative().derivative()(t)
        ay = qr_p.quad_traj['y'].piece_poly.derivative().derivative()(t)
        az = qr_p.quad_traj['z'].piece_poly.derivative().derivative()(t)

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

        # Create a piecewise polynomial for n_laps with entry and exit (if they exist) - for animation and sending trajectory
        self.ppoly_laps = utils.create_laps_trajectory(self.n_laps,
                                                  self.qr_polytraj,
                                                  self.qr_p_entry,
                                                  self.qr_p_exit,
                                                  self.entry_ID,
                                                  self.exit_ID,
                                                  self.qr_polytraj.closed_loop)


    def on_disc_updated_signal(self):
        if (self.global_dict['disc_out_waypoints'] is None or
                    'x' not in self.global_dict['disc_out_waypoints'].keys() or
                    'y' not in self.global_dict['disc_out_waypoints'].keys() or
                    'z' not in self.global_dict['disc_out_waypoints'].keys()):
            print("Spurious callback in minsnap; no discretized waypoints")
            return

        waypoints = self.global_dict['disc_out_waypoints']
        costs = dict()
        costs['x'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['y'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['z'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['yaw'] = [0, 0, 1]  # minimum acceleration
        order=dict(x=9, y=9, z=9, yaw=5)
        waypoints['yaw'] = np.array([0.0] * np.size(waypoints['x']))
        self.qr_polytraj = quadrotor_polytraj.QRPolyTraj(waypoints,
                                self.time_penalty,
                                costs=costs,
                                order=order,
                                #seed_times = np.array([1.0] * (np.size(waypoints['x']) - 1)),
                                seed_avg_vel = 1.0,
                                face_front = np.array([False] +
                                    [True] * (np.size(waypoints['x']) - 2) + [False]))

        self.exit_ID = self.qr_polytraj.n_seg
        self.interactive_marker_worker.exit_ID = self.exit_ID

        self.update_path_markers()
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp=acc_wp)

    def on_optimize_button_click(self):
        if self.qr_polytraj is None:
            print("No qr_polytraj")
            return
        self.start_time = time.time()
        # copy time penalty, which is updated from input box
        self.qr_polytraj.time_penalty = self.time_penalty

        self.show_obstacle_plan_updates = False

        self.controls_enabled = False
        self.optimize_button.setEnabled(False)
        self.stop_optimize_button.setEnabled(True)

        self.optimizer_thread = QThread()
        self.optimizer_thread.app = self

        self.optimizer_worker = OptimizerWorker(self.on_optimizer_worker_done)
        self.optimizer_worker.local_qr_polytraj = self.qr_polytraj
        self.optimizer_worker.moveToThread(self.optimizer_thread)
        self.optimizer_thread.started.connect(self.optimizer_worker.task)
        self.optimizer_thread.start()

    def on_animate_radio_button_toggle(self, animate):

        if animate:
            print("start animation")
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

    def on_close_loop_checkbox_toggle(self, close_loop):
        if close_loop:
            # CLose loop activated
            print("Closing loop")
            self.qr_polytraj.close_trajectory_loop()
        else:
            # Opening loop
            print("Opening loop")
            self.qr_polytraj.open_trajectory_loop()


        self.update_path_markers()
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints, closed_loop=self.qr_polytraj.closed_loop)
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints, closed_loop=self.qr_polytraj.closed_loop,acc_wp=acc_wp)


    def on_stop_optimize_button_click(self):
        self.optimizer_worker.stop()
        self.optimizer_thread.quit()
        self.stop_optimize_button.setEnabled(False)

    def on_send_trajectory_button_click(self):
        if self.qr_polytraj is None or self.ppoly_laps is None:
            print("Not sending; polytraj is None")
            return
        msg = PolyTraj_msg()

        self.ppoly_laps = utils.create_laps_trajectory(self.n_laps,
                                                  self.qr_polytraj,
                                                  self.qr_p_entry,
                                                  self.qr_p_exit,
                                                  self.entry_ID,
                                                  self.exit_ID,
                                                  self.qr_polytraj.closed_loop)
                # else:
                # self.ppoly_laps = utils.create_TO_traj_LAND_trajectory(self.qr_polytraj, self.qr_p_entry, self.qr_p_exit)

        # Set yaw to be along the trajectory
        if self.set_yaw:
            self.set_yaw_to_traj_vel()

        utils.fill_n_laps_poly_traj_msg(self.ppoly_laps,msg)


        self.polytraj_pub.publish(msg)

    def on_animate_eval_callback(self, t):
        if self.qr_polytraj is None:
            return

        if self.ppoly_laps is None:
            self.ppoly_laps = utils.create_laps_trajectory(self.n_laps,
                                                      self.qr_polytraj,
                                                      self.qr_p_entry,
                                                      self.qr_p_exit,
                                                      self.entry_ID,
                                                      self.exit_ID,
                                                      self.qr_polytraj.closed_loop)



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
        # print("R is {}".format(R))
        return (t_max, x, y, z, q[0], q[1], q[2], q[3])

    def on_waypoint_control_callback(self, position, index, qr_type):

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

        if qr_p.closed_loop:
            if index == qr_p.n_seg: # Last waypoint
                index = 0 # move the first waypoint instead
            if index == 0:
                # If first point being updated, also update last
                qr_p.update_xyz_yaw_partial_waypoint(qr_p.n_seg,position,
                                                                 defer=True)


        qr_p.update_xyz_yaw_partial_waypoint(index,position,defer=False)
        # TODO(mereweth@jpl.nasa.gov) - return which waypoints/yaws changed by
        # more than some threshold?
        qr_p.set_yaw_des_from_traj()

        # Up to the user to know which controls they want to update
        # In this case, some yaws have changed but not been updated. This is
        # a performance tradeoff - see above
        acc_wp = self.get_accel_at_waypoints(qr_type)

        if qr_type is "main":
            self.interactive_marker_worker.update_controls(qr_p.waypoints,
                                                           indices=[index],closed_loop=qr_p.closed_loop,acc_wp=acc_wp)
            self.qr_polytraj = qr_p
        elif qr_type is "entry":
            self.interactive_marker_worker_entry.update_controls(qr_p.waypoints,
                                                           indices=[index],closed_loop=qr_p.closed_loop,acc_wp=acc_wp)
            self.qr_entry = qr_p
        elif qr_type is "exit":
            self.interactive_marker_worker_exit.update_controls(qr_p.waypoints,
                                                           indices=[index],closed_loop=qr_p.closed_loop,acc_wp=acc_wp)
            self.qr_exit = qr_p

        self.update_path_markers(qr_type=qr_type)

        self.laps_set_flag = False

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

    def on_optimizer_worker_done(self):
        self.qr_polytraj = self.optimizer_worker.local_qr_polytraj

        # Reuse functionality to destroy optimizer thread
        self.on_stop_optimize_button_click()

        print("times are {}".format(self.qr_polytraj.times))
        print("time ratios are {}".format(self.qr_polytraj.times/np.max(self.qr_polytraj.times)))

        map = self.check_for_map()

        if map is None:
            run_obstacles = False
            collision_segs = np.array([])
        else:
            samp_mult = 500

            # find the segments in collision
            collision_segs = self.find_segments_in_collision(samp_mult=samp_mult)
            print("Collision segs {}, size {}".format(collision_segs,np.shape(collision_segs)))
            run_obstacles = True
        # try:
        #     map = self.global_dict['fsp_out_map']
        #     run_obstacles = True
        # except AttributeError:
        #     print("Not considering collisions Need to have loaded an ESDF")
        #     run_obstacles = False
        # Check for collision
        # IF collisions, then plan with obstacles
        if np.size(collision_segs) != 0 and run_obstacles:
            self.on_plan_with_obstacles_button_click()

            self.optimizer_thread = QThread()
            self.optimizer_thread.app = self

            self.optimizer_worker = OptimizerWorker(self.on_optimizer_worker_done)
            self.optimizer_worker.local_qr_polytraj = self.qr_polytraj
            self.optimizer_worker.moveToThread(self.optimizer_thread)
            self.optimizer_thread.started.connect(self.optimizer_worker.task)
            self.optimizer_thread.start()
        else:
            self.qr_polytraj.outer_opt_time = time.time() - self.start_time

            self.optimize_button.setEnabled(True)
            self.controls_enabled = True
            self.update_path_markers()
            acc_wp = self.get_accel_at_waypoints("main")
            self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

            self.show_obstacle_plan_updates = True

    def on_load_trajectory_button_click(self, checked=False, filename=None):
        if filename is None:
            filename = QFileDialog.getOpenFileName(self,
                                                   'Import trajectory', #path,
                                                   "Traj pickle (*.traj)")
            if filename and len(filename)>0:
                filename = filename[0]
                #self.traj_file_label.setText( filename )
            else:
                print("Invalid file path")
                return

        try:
            with open(filename, 'rb') as f:
                self.qr_polytraj = pickle.load(f)
                self.qr_polytraj.make_forwards_compatible()
            print("Loaded trajectory from {}".format(filename))

        except Exception as e:
            print("Could not load pickled QRPolyTraj from {}".format(filename))
            print(e.message)
            return

        if self.qr_polytraj.closed_loop is False:
            print("setting exit ID")
            self.exit_ID = self.qr_polytraj.n_seg
            self.interactive_marker_worker.exit_ID = self.exit_ID

        self.update_path_markers()

        #TODO(mereweth@jpl.nasa.gov) - figure out where this is necessary
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

    def on_save_trajectory_button_click(self, checked=False, filename=None):
        if self.qr_polytraj is None:
            print("Nothing to save")
            return

        if filename is None:
            filename = QFileDialog.getSaveFileName(self, 'Save trajectory', #path,
                                                   "Trajectory files (*.traj)")
            if filename and len(filename)>0:
                filename = filename[0]
                #self.traj_file_label.setText( filename )
            else:
                print("Invalid file path")
                return
        try:
            with open(filename, 'wb') as f:
                print("Saving pickled QRPolyTraj to {}".format(filename))
                pickle.dump(self.qr_polytraj, f, 2 )
        except Exception as e:
            print("Could not save pickled QRPolyTraj to {}".format(filename))
            print(e.message)

    def on_load_quad_parameters(self,checked=False, filename=None):
        if filename is None:
            filename = QFileDialog.getOpenFileName(self,
                                                   'Import quad params', #path,
                                                   "params (*.yaml)")
            if filename and len(filename)>0:
                filename = filename[0]
                #self.traj_file_label.setText( filename )
            else:
                print("Invalid file path")
                return

        try:
            with open(filename, 'rb') as f:
                self.quad_params = controls.load_params(filename)
            print("Loaded quad parameters from {}".format(filename))

        except Exception as e:
            print("Could not load quad parameters from {}".format(filename))
            print(e.message)
            return

    def on_check_maximum_performance(self, samp_mult = 100):
        """
        Check maximum accel, and rpm
        """

        if self.quad_params is None:
            return

        # samp_mult = samp_mult*self.qr_polytraj.n_seg
        trans_times = utils.seg_times_to_trans_times(self.qr_polytraj.times)
        t_total = trans_times[-1] - trans_times[0]
        t_vec = np.linspace(trans_times[0], trans_times[-1], t_total*samp_mult)

        # print trans_times
        # print t_total
        # print t_vec

        # Initilize maximums
        a_max = 0.0
        rpm_max = 0.0
        thr_max = 0.0
        vel_max = 0.0
        M_max = 0.0

        count = 0

        self.rpm_max_all = [0 for x in range(len(trans_times)-1)]
        self.rpm_min_all = [900000 for x in range(len(trans_times)-1)]
        self.vel_max_all = [0 for x in range(len(trans_times)-1)]
        self.acc_max_all = [0 for x in range(len(trans_times)-1)]
        self.thrust_max_all = [0 for x in range(len(trans_times)-1)]

        current_seg = 1
        for t in t_vec:
            if t > trans_times[current_seg]:
                current_seg =  current_seg + 1

            vel = np.array([self.qr_polytraj.quad_traj['x'].piece_poly.derivative()(t),
            self.qr_polytraj.quad_traj['y'].piece_poly.derivative()(t),
            self.qr_polytraj.quad_traj['z'].piece_poly.derivative()(t)])
            v_mag = np.linalg.norm(vel)

            accel = np.array([self.qr_polytraj.quad_traj['x'].piece_poly.derivative().derivative()(t),
            self.qr_polytraj.quad_traj['y'].piece_poly.derivative().derivative()(t),
            self.qr_polytraj.quad_traj['z'].piece_poly.derivative().derivative()(t)])
            a_mag = np.linalg.norm(accel)

            jerk = np.array([self.qr_polytraj.quad_traj['x'].piece_poly.derivative().derivative().derivative()(t),
            self.qr_polytraj.quad_traj['y'].piece_poly.derivative().derivative().derivative()(t),
            self.qr_polytraj.quad_traj['z'].piece_poly.derivative().derivative().derivative()(t)])

            snap = np.array([self.qr_polytraj.quad_traj['x'].piece_poly.derivative().derivative().derivative().derivative()(t),
            self.qr_polytraj.quad_traj['y'].piece_poly.derivative().derivative().derivative().derivative()(t),
            self.qr_polytraj.quad_traj['z'].piece_poly.derivative().derivative().derivative().derivative()(t)])

            yaw = self.qr_polytraj.quad_traj['yaw'].piece_poly(t)
            yaw_dot = self.qr_polytraj.quad_traj['yaw'].piece_poly.derivative()(t)
            yaw_ddot = self.qr_polytraj.quad_traj['yaw'].piece_poly.derivative().derivative()(t)

            # Get rotation matrix
            R, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix')

            # Thrust
            thrust, thrust_mag = body_frame.get_thrust(accel)

            # Angular rates
            ang_vel = angular_rates_accel.get_angular_vel(thrust_mag,jerk,R,yaw_dot)

            # Angular accelerations
            ang_accel = angular_rates_accel.get_angular_accel(thrust_mag,jerk,snap,R,ang_vel,yaw_ddot)

            if self.quad_params is not None:
                # torques
                torques = controls.get_torques(ang_vel, ang_accel, self.quad_params)

                # Get rotor speeds
                rpm = controls.get_rotor_speeds(torques,thrust_mag*self.quad_params['mass'],self.quad_params)
            else:
                print("no max rpm as quad parameters are needed")
                torques = np.zeros(3)
                rpm = np.zeros(4)

            max_torque = np.max(torques)
            max_rpm = np.max(rpm)
            min_rpm = np.min(rpm)

            if thrust_mag > thr_max:
                thr_max = thrust_mag
            if a_mag > a_max:
                a_max = np.linalg.norm(accel)
            if max_rpm > rpm_max:
                rpm_max = max_rpm
            if max_torque > M_max:
                M_max = max_torque
            if v_mag > vel_max:
                vel_max = v_mag

            if max_rpm > self.rpm_max_all[current_seg-1]:
                self.rpm_max_all[current_seg-1] = max_rpm
            if min_rpm < self.rpm_min_all[current_seg-1]:
                self.rpm_min_all[current_seg-1] = min_rpm
            if v_mag > self.vel_max_all[current_seg-1]:
                self.vel_max_all[current_seg-1] = v_mag
            if a_mag > self.acc_max_all[current_seg-1]:
                self.acc_max_all[current_seg-1] = a_mag
            if thrust_mag > self.thrust_max_all[current_seg-1]:
                self.thrust_max_all[current_seg-1] = thrust_mag

            count += 1

        self.a_max = a_max
        self.rpm_max = rpm_max
        self.M_max = M_max
        self.v_max = vel_max
        self.thr_max = thr_max

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
        yaw = transforms3d.euler.quat2euler(q,'rzyx')[0]

        self.update_first_waypoint( position, yaw, defer=True)
        self.update_last_waypoint( position, yaw, defer=False)
        self.t=0 #reset visualization

        self.update_path_markers()

        #TODO(mereweth@jpl.nasa.gov) - figure out where this is necessary
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

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
        yaw = transforms3d.euler.quat2euler(q,'rzyx')[0]
        # yaw = self.qr_polytraj.waypoints['yaw'][0,0]

        self.update_first_waypoint( position, yaw, defer=True)
        position['z'] += 0.0
        # yaw = self.qr_polytraj.waypoints['yaw'][0,-1]
        self.update_last_waypoint( position, yaw, defer=False)
        self.t=0 #reset visualization

        self.update_path_markers()

        #TODO(mereweth@jpl.nasa.gov) - figure out where this is necessary
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

    def on_add_waypoint_button_click(self):

        if self.qr_polytraj is None:
            print("No Waypoints received yet. Need to initialise")
            return

        if self.index_waypoint > self.qr_polytraj.n_seg+1:
            # Index too large
            print("Index is too large, not adding")
            return

        # Load from GUI
        new_waypoint = dict()
        new_waypoint['x'] = self.new_x_waypoint
        new_waypoint['y'] = self.new_y_waypoint
        new_waypoint['z'] = self.new_z_waypoint
        new_waypoint['yaw'] = self.new_yaw_waypoint
        index = self.index_waypoint

        new_der_fixed=dict(x=True,y=True,z=True,yaw=True)
        # TODO (bmorrell@jpl.nasa.gov) have der_Fixed as an input

        # Insert waypoint
        self.insert_waypoint(new_waypoint, index, new_der_fixed,defer=False)

        self.update_path_markers()

        #TODO(mereweth@jpl.nasa.gov) - figure out where this is necessary
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

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

        # Delete waypoint
        self.delete_waypoint(index, defer=False)

        self.update_path_markers()

        #TODO(mereweth@jpl.nasa.gov) - figure out where this is necessary
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)


    def update_waypoint(self,position, yaw, index, defer=False):
        xyz_yaw_partial_waypoint = dict()

        for key in position.keys():
            if np.isnan(position[key]):
                print("nan detected in update_waypoint")
                return
            xyz_yaw_partial_waypoint[key] = position[key]
        xyz_yaw_partial_waypoint["yaw"] = yaw

        self.qr_polytraj.update_xyz_yaw_partial_waypoint(index, xyz_yaw_partial_waypoint, defer=defer)

        self.qr_polytraj.do_update_trajectory_markers = True
        self.qr_polytraj.do_update_control_yaws = True
        self.laps_set_flag = False

    def update_first_waypoint(self,position, yaw, defer=False):
        self.update_waypoint(position, yaw, 0, defer=defer)

    def update_last_waypoint(self,position, yaw, defer=False):
        self.update_waypoint(position, yaw, self.qr_polytraj.n_seg, defer=defer)

    def insert_waypoint(self, new_waypoint, index, new_der_fixed,defer=False, qr_type="main"):

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

        # TODO (bmorrell@jpl.nasa.gov) have the times as an input
        print("inserting at {}\n waypoint {}\n times {}\n der_fixed, {}".format(index,new_waypoint, new_times, new_der_fixed ))
        qr_p.insert(index, new_waypoint, new_times, new_der_fixed, defer=defer)

        qr_p.do_update_trajectory_markers = True
        qr_p.do_update_control_yaws = True

        if qr_type is "main":
            self.qr_polytraj = qr_p
        elif qr_type is "entry":
            self.qr_p_entry = qr_p
        elif qr_type is "exit":
            self.qr_p_exit = qr_p

        self.laps_set_flag = False

    def delete_waypoint(self, index, defer=False, qr_type="main"):

        if qr_type is "main":
            qr_p = self.qr_polytraj
        elif qr_type is "entry":
            qr_p = self.qr_p_entry
        elif qr_type is "exit":
            qr_p = self.qr_p_exit

        if index > qr_p.n_seg + 1:
            return

        # Times. Current to just double the time for adjacent segment(s)
        if index == 0:
            # Double first segment
            new_time = qr_p.times[index]
        elif index == qr_p.times.size:
            # Double last segment
            new_time = qr_p.times[index-1]
        else:
            # Add adjacent segments
            new_time = qr_p.times[index-1] + qr_p.times[index]

        # TODO (bmorrell@jpl.nasa.gov) have the times as an input.
        # TODO (bmorrell@jpl.nasa.gov) Look at having der_fixed as an input

        qr_p.delete(index, new_time, defer=defer)

        qr_p.do_update_trajectory_markers = True
        qr_p.do_update_control_yaws = True

        if qr_type is "main":
            self.qr_polytraj = qr_p
        elif qr_type is "entry":
            self.qr_p_entry = qr_p
        elif qr_type is "exit":
            self.qr_p_exit = qr_p

        self.laps_set_flag = False

    def on_update_times_button_clicked(self):
        """
        Update times
        """
        times = self.qr_polytraj.times

        index = self.time_edit_index

        edit_time = self.edit_time

        times[index] = edit_time

        self.qr_polytraj.update_times(times)

        self.update_path_markers("main")
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

        # waypoints = dict()
        # for key in self.qr_polytraj.waypoints.keys():
        #     waypoints[key] = self.qr_polytraj.waypoints[key][:,0]
        #
        # self.qr_polytraj.update_xyz_yaw_partial_waypoint(0,waypoints)

    def on_scale_total_time_button_clicked(self):
        """
        Update time for the complete trajectory
        """
        if self.total_time_setting is None:
            print("Need to input time")
            return
        times = self.qr_polytraj.times

        times = times/np.sum(times)*self.total_time_setting

        self.qr_polytraj.update_times(times)

        self.update_path_markers("main")
        acc_wp = self.get_accel_at_waypoints("main")
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

    def compute_nearest_distance_to_obstacles(self):

        try:
            map = self.global_dict['fsp_out_map']
        except AttributeError:
            print("Not computing nearest distance. Need to have loaded an ESDF")
            return

        if self.qr_polytraj.nodes is None:
            waypoints = self.qr_polytraj.waypoints
        else:
            # If nodes have already been initialised, use those to compute the distances
            waypoints = self.qr_polytraj.nodes
        n_seg = self.qr_polytraj.n_seg
        n_samples = 100 # TODO (bmorrell@jpl.nasa.gov) Make this a setting in qr_polytraj
        l_max = np.zeros(n_seg)

        for k in range(0,n_seg):
            # Create query points
            query = np.matrix(np.zeros([3,n_samples]),dtype='double')
            dist = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='double')
            obs = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='int32')

            # load in x, y, z points
            query[0,:] = np.linspace(waypoints['x'][0,k],waypoints['x'][0,k+1],n_samples)
            query[1,:] = np.linspace(waypoints['y'][0,k],waypoints['y'][0,k+1],n_samples)
            query[2,:] = np.linspace(waypoints['z'][0,k],waypoints['z'][0,k+1],n_samples)

            # Query the database
            map.getDistanceAtPosition(query, dist, obs)

            # Find the minimum obstacle distance
            l_max[k] = np.min(dist)

        # Update l_max in the qr_polytraj object
        self.qr_polytraj.l_max = l_max

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
        entry_waypoints = utils.form_waypoints_polytraj(entry_waypoints,self.qr_polytraj.order)
        entry_waypoints, entry_der_fixed = utils.form_entry_or_exit_waypoints("entry",self.qr_polytraj, entry_waypoints, entry_ID, add_on_der_fixed=dict(x=None,y=None,z=None,yaw=None))

        # Create new qr_polytraj
        self.qr_p_entry = quadrotor_polytraj.QRPolyTraj(entry_waypoints, self.qr_polytraj.time_penalty, der_fixed=entry_der_fixed, costs=self.qr_polytraj.costs,order=self.qr_polytraj.order,
                                            closed_loop=False, seed_times=entry_times, face_front=[False]*entry_waypoints['x'].shape[1])

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
        exit_waypoints = utils.form_waypoints_polytraj(exit_waypoints,self.qr_polytraj.order)
        exit_waypoints, exit_der_fixed = utils.form_entry_or_exit_waypoints("exit",self.qr_polytraj, exit_waypoints, exit_ID, add_on_der_fixed=add_on_der_fixed)

        # Create new qr_polytraj
        self.qr_p_exit = quadrotor_polytraj.QRPolyTraj(exit_waypoints, self.qr_polytraj.time_penalty, der_fixed=exit_der_fixed, costs=self.qr_polytraj.costs,order=self.qr_polytraj.order,
                                            closed_loop=False, seed_times=exit_times, face_front=[False]*exit_waypoints['x'].shape[1])

        # Create Controls and trajectory for display
        print("forming landing")
        self.update_path_markers(qr_type="exit")
        self.interactive_marker_worker_exit.make_controls(self.qr_p_exit.waypoints)
        acc_wp = self.get_accel_at_waypoints("exit")
        self.interactive_marker_worker_exit.update_controls(self.qr_p_exit.waypoints,acc_wp=acc_wp)

    def on_add_takeoff_button_click(self):
        self.add_take_off()

    def on_add_landing_button_click(self):
        self.add_landing()

    def update_entry_point(self):
        if self.qr_p_entry is None:
            return

        entry_ID = self.entry_ID

        # Get new state at the waypoint
        new_waypoint = utils.get_state_at_waypoint(self.qr_polytraj, entry_ID)
        index = self.qr_p_entry.waypoints['x'].shape[1]-1

        # Update trajectory
        self.qr_p_entry.update_xyz_yaw_partial_waypoint(index,new_waypoint,defer=False)

        self.update_path_markers(qr_type='entry')

    def update_exit_point(self):
        if self.qr_p_exit is None:
            return

        exit_ID = self.exit_ID

        # Get new state at the waypoint
        new_waypoint = utils.get_state_at_waypoint(self.qr_polytraj, exit_ID)
        index = 0

        # Update trajectory
        self.qr_p_exit.update_xyz_yaw_partial_waypoint(index,new_waypoint,defer=False)

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

        self.update_path_markers()
        # # Create a piecewise polynomial for n_laps with entry and exit
        # self.ppoly_laps = utils.create_laps_trajectory(self.n_laps,
        #                                           self.qr_polytraj,
        #                                           self.qr_p_entry,
        #                                           self.qr_p_exit,
        #                                           self.entry_ID,
        #                                           self.exit_ID,
        #                                           self.qr_polytraj.closed_loop)
        #
        # # Set yaw to be along the trajectory
        # if self.set_yaw:
        #     self.set_yaw_to_traj_vel()


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
                exit_waypoints[key] = np.array([position[key]+0.4,position[key]+0.2,position[key]-0.1])
            elif key is 'yaw':
                exit_waypoints[key] = np.array([self.qr_polytraj.waypoints[key][0,self.exit_ID]]*3)
            else:
                exit_waypoints[key] = np.array([position[key]]*3)

            n_der = self.qr_polytraj.der_fixed[key].shape[0]
            add_on_der_fixed[key] = np.ones((n_der,3),dtype=bool)

        # Set for a long final time to descend slowly
        # exit_times = np.array([2.0,1.0,10.0])
        exit_times = np.array([2*self.exit_time/3,self.exit_time/3,10.0])

        print(exit_waypoints['x'].shape)

        self.add_landing(exit_waypoints, exit_times, add_on_der_fixed)

    def on_create_n_laps_button_click(self):

        if not self.qr_polytraj.closed_loop:
            print("Need to form closed loop")
            return

        # Create a piecewise polynomial for n_laps with entry and exit
        self.ppoly_laps = utils.create_laps_trajectory(self.n_laps,
                                                  self.qr_polytraj,
                                                  self.qr_p_entry,
                                                  self.qr_p_exit,
                                                  self.entry_ID,
                                                  self.exit_ID,
                                                  self.qr_polytraj.closed_loop)

        # Set yaw to be along the trajectory
        if self.set_yaw:
            self.set_yaw_to_traj_vel()


        self.laps_set_flag = True

        self.update_path_markers()

        print("Ppoly created for {} laps. Times are:\n{}".format(self.n_laps,self.ppoly_laps['x'].x))

    def get_accel_at_waypoints(self,qr_type="main"):
        if self.qr_polytraj is None:
            return

        if qr_type is "main":
            qr_p = self.qr_polytraj
            t = utils.seg_times_to_trans_times(qr_p.times)
            ax = qr_p.quad_traj['x'].piece_poly.derivative().derivative()(t)
            ay = qr_p.quad_traj['y'].piece_poly.derivative().derivative()(t)
            az = qr_p.quad_traj['z'].piece_poly.derivative().derivative()(t)
            return np.reshape(np.array([ax,ay,az]),(3,np.size(ax)))
        elif qr_type is "entry":
            # qr_p = self.qr_p_entry
            return None
        elif qr_type is "exit":
            # qr_p = self.qr_p_exit
            return None

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

    def check_distance_to_trajectory(self,x=None,y=None,z=None,samp_mult=100):
        map = self.check_for_map()
        if map is None:
            return

        # TODO Check collisions for the take-off and landing trajectories

        if x is None:
            # Generate x, y, z points
            trans_times = utils.seg_times_to_trans_times(self.qr_polytraj.times)
            t_total = trans_times[-1] - trans_times[0]
            t = np.linspace(trans_times[0], trans_times[-1], t_total*samp_mult)

            x = self.qr_polytraj.quad_traj['x'].piece_poly(t)
            y = self.qr_polytraj.quad_traj['y'].piece_poly(t)
            z = self.qr_polytraj.quad_traj['z'].piece_poly(t)

        # Create query points
        query = np.matrix(np.zeros([3,x.size]),dtype='double')
        dist = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='double')
        obs = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='int32')

        # load in x, y, z points
        query[0,:] = np.around(x,4)
        query[1,:] = np.around(y,4)
        query[2,:] = np.around(z,4)

        # Query the database
        map.getDistanceAtPosition(query, dist, obs)

        # Add buffer on quad:
        dist -= self.quad_buffer

        if self.unknown_is_free:
            dist[obs != 1] = 2.0

        # return the distance
        return dist

    def find_segments_in_collision(self,samp_mult=100):

        trans_times = utils.seg_times_to_trans_times(self.qr_polytraj.times)

        collision = np.zeros(self.qr_polytraj.n_seg,dtype=bool)

        min_dist = np.zeros(self.qr_polytraj.n_seg)

        for i in range(self.qr_polytraj.n_seg):
            # Generate x, y, z points
            t = np.linspace(trans_times[i], trans_times[i+1], samp_mult)

            x = self.qr_polytraj.quad_traj['x'].piece_poly(t)
            y = self.qr_polytraj.quad_traj['y'].piece_poly(t)
            z = self.qr_polytraj.quad_traj['z'].piece_poly(t)

            dist = self.check_distance_to_trajectory(x,y,z,samp_mult=samp_mult)

            if np.min(dist)<0.0:
                collision[i] = True
            min_dist[i] = np.min(dist)
            self.min_dist = min_dist

        return np.where(collision)[0]

    def set_yaw_to_traj_vel(self):

        if self.ppoly_laps is None:
            return
        # trans_times = utils.seg_times_to_trans_times(self.ppoly_laps.x)
        trans_times = self.ppoly_laps['x'].x
        t_total = trans_times[-1] - trans_times[0]
        t = np.linspace(trans_times[0], trans_times[-1], t_total*1000)

        yaw = np.arctan2(self.ppoly_laps['y'].derivative()(t),self.ppoly_laps['x'].derivative()(t))

        spl = sp.interpolate.splrep(t,yaw)
        self.ppoly_laps['yaw'] = sp.interpolate.PPoly.from_spline(spl)

        # self.qr_polytraj.quad_traj['yaw'].piece_poly = np.interpolate.PPoly.from_spline(spl)


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
            trans_times = utils.seg_times_to_trans_times(self.qr_polytraj.times)
            t_total = trans_times[-1] - trans_times[0]
            t = np.linspace(trans_times[0], trans_times[-1], t_total*100)

            x = self.qr_polytraj.quad_traj['x'].piece_poly(t)
            y = self.qr_polytraj.quad_traj['y'].piece_poly(t)
            z = self.qr_polytraj.quad_traj['z'].piece_poly(t)

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

    def get_max_violation_points(self,dist_traj,grad_traj):

        tp = np.array([])
        grad_tol = 1e-10
        dist_tol = 0.2

        for i in range(1,grad_traj.size-1):
            if grad_traj[i] < grad_tol:
                if np.sign(grad_traj[i-1]) != np.sign(grad_traj[i+1]) and grad_traj[i+1] != 0:
                    if dist_traj[i] < dist_tol:
                        tp = np.append(tp,i)

        return tp

    def update_results_display(self, max_a=-1.0):
        """
        Set text for comp time and iterations to display in the GUI
        """
        self.outer_time_line_edit.setText(str(self.qr_polytraj.outer_opt_time))
        comp_time = 0.0
        for key in self.qr_polytraj.quad_traj.keys():
            comp_time += self.qr_polytraj.quad_traj[key].opt_time
        self.comp_time_line_edit.setText(str(comp_time))
        # self.comp_time_line_edit.setText(str(self.qr_polytraj.opt_time))
        self.traj_time_line_edit.setText(str(np.sum(self.qr_polytraj.times)))

        # self.on_check_maximum_performance()
        # if self.a_max > 0.0:
        # self.traj_accel_line_edit.setText(str(self.a_max))
        # else:
        self.traj_accel_line_edit.setText(str(max_a))
        # print("\nMaximum performance:\n\nMax RPM: {}\nMax M {}\nMax Thr: {}\nMax A: {}\nMax V: {}\n\n".format(self.rpm_max,self.M_max,self.thr_max,self.a_max,self.v_max))
        # print("Astro cost is: {}".format(self.qr_polytraj.data_track.cost[-1]))

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

    def add_waypoint_in_segment(self,seg_index=3):

        if self.qr_polytraj is None:
            return

        if self.global_dict['full_waypoints'] is None:
            print("can't plan with obstacles. NEED waypoints set from DISC")
            return

        mask = self.global_dict['disc_mask']

        # Indices in the full waypoints for the waypoints on either side of the segment
        start_index = mask[seg_index]
        end_index = mask[seg_index+1]

        new_index = np.round((start_index+end_index)/2)
        print("new_index is {}".format(new_index))

        new_waypoint = dict()
        new_waypoint['x'] = self.global_dict['full_waypoints']['x'][new_index]
        new_waypoint['y'] = self.global_dict['full_waypoints']['y'][new_index]
        new_waypoint['z'] = self.global_dict['full_waypoints']['z'][new_index]
        # TODO (bmorrell@jpl.nasa.gov) check yaw
        if self.global_dict['full_waypoints']['yaw'].size != 0:
            new_waypoint['yaw'] = self.global_dict['full_waypoints']['yaw'][new_index]
        else:
            new_waypoint['yaw'] = 0.0

        new_der_fixed=dict(x=True,y=True,z=True,yaw=True)

        # Insert new waypoint
        self.insert_waypoint(new_waypoint, seg_index+1, new_der_fixed,defer=False,qr_type="main")

        # Update the global dictionary with the new waypoint selections
        self.global_dict['disc_mask'] = np.insert(mask,seg_index+1,new_index)

        # TODO: (bmorrell@jpl.nasa.gov) Do we need to do this?
        self.global_dict['disc_out_waypoints'] = self.qr_polytraj.waypoints



    def on_plan_with_obstacles_button_click(self):

        # Check if an ESDF is loaded
        if ('fsp_out_map' in self.global_dict.keys() and
                        self.global_dict['fsp_out_map'] is not None):
            print('Using ESDF')
        else:
            print('Error: Need to load an ESDF before planning in restricted freespace')
            return

        if self.qr_polytraj is None:
            return


        samp_mult = 500

        # find the segments in collision
        collision_segs = self.find_segments_in_collision(samp_mult=samp_mult)

        iteration = 0

        while np.shape(collision_segs)[0] != 0:
            print("Segments in collision are: {}".format(collision_segs))

            # TODO (bmorrell@jpl.nasa.gov) review logic of selecting which segment to change
            self.add_waypoint_in_segment(collision_segs[0])
            iteration += 1
            print("interation {}".format(iteration))

            # find the segments in collision
            collision_segs = self.find_segments_in_collision(samp_mult=samp_mult)

            if iteration>50:
                return

            if self.show_obstacle_plan_updates:
                self.update_path_markers()
                acc_wp = self.get_accel_at_waypoints()
                self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
                self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

        print("Done planning with obstacles in {} iterations".format(iteration))
        # return True

    def on_set_yaw_to_random_button_click(self):
        """
        To set randomized yaw values for sys ID flights
        """

        if self.qr_polytraj is None:
            return

        print("Setting yaw to random")
        # Set yaw to random, based on the input delta
        self.qr_polytraj.set_yaw_to_random(self.yaw_rand_delta)

        self.update_path_markers()
        acc_wp = self.get_accel_at_waypoints()
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)

    def on_set_yaw_to_constant_button_click(self):
        """
        To set randomized yaw values for sys ID flights
        """

        if self.qr_polytraj is None:
            return

        print("Setting yaw to random")
        # Set yaw to random, based on the input delta
        self.qr_polytraj.set_yaw_to_constant(self.yaw_constant)

        self.update_path_markers()
        acc_wp = self.get_accel_at_waypoints()
        self.interactive_marker_worker.make_controls(self.qr_polytraj.waypoints)
        self.interactive_marker_worker.update_controls(self.qr_polytraj.waypoints,acc_wp = acc_wp)




def main():
    from torq_gcs.plan.unco import QRPolyTrajGUI

    app = QApplication( sys.argv )

    global_dict = dict()
    qrp_gui = QRPolyTrajGUI(global_dict, new_node=True)

    try:
        import imp
        conf = imp.load_source('torq_config',
                        os.path.abspath('~/Desktop/environments/344.py'))
        conf.torq_config(qrp_gui)
    except Exception as e:
        print("Non fatal error in config script")
        print(e)

    qrp_gui.show()

    return app.exec_()

if __name__=="__main__":
    sys.exit(main())
