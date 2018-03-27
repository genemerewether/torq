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

from minsnap import utils, quadrotor_polytraj
from diffeo import body_frame

import rospkg, rospy, tf
from python_qt_binding import loadUi
from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *
from python_qt_binding.QtWidgets import *

#from torq_gcs import plan
from torq_gcs.plan.ros_helpers import TrajectoryDisplayWorker
from torq_gcs.plan.ros_helpers import WaypointControlWorker
from torq_gcs.plan.ros_helpers import NodePathWorker
from torq_gcs.plan.ros_helpers import AnimateWorker

from px4_msgs.msg import PolyTraj as PolyTraj_msg
from geometry_msgs.msg import PoseStamped

class OptimizerWorker(QObject):
    finished = Signal() # class variable shared by all instances

    def __init__(self, finished_callback, parent=None):
        super(OptimizerWorker, self).__init__(parent)
        self.finished.connect(finished_callback)
        self.is_running = True

    def task(self):
        print("Optimizing in background thread")
        self.local_qr_polytraj.update_times(self.local_qr_polytraj.times)
        self.finished.emit()

    def stop(self):
        self.is_running = False

class FreespacePlannerGUI(QWidget):
    def __init__(self,
                 global_dict,
                 node_name='taco',
                 new_node=False,
                 parent=None):
        super(FreespacePlannerGUI, self).__init__(parent)
        ui_file = os.path.join(rospkg.RosPack().get_path('torq_gcs'),
                               'resource', 'plan', 'FreespacePlan.ui')
        loadUi(ui_file, self)

        self.qr_polytraj = None
        self.time_penalty = 1000
        self.A_max = 12.0
        self.A_sel_ind = 0
        self.quad_buffer = 0.15

        self.unknown_is_free = True

        self.n_samp_node_path = 2
        # self.distance_current = False # Flag to say if current store distance values are current
        self.plot_traj_color = True
        self.plot_path_color = True

        self.last_pose_time   = None
        self.last_position    = None
        self.last_orientation = None

        self.controls_enabled = True

        self.new_x_waypoint = 0.
        self.new_y_waypoint = 0.
        self.new_z_waypoint = 0.
        self.new_yaw_waypoint = 0.
        self.index_waypoint = 0

        if new_node:
            rospy.init_node(node_name, anonymous=True)

        # NOTE (mereweth@jpl.nasa.gov) - this still does not bypass the GIL
        # unless rospy disables the GIL in C/C++ code
        self.ros_helper_thread = QThread()
        self.ros_helper_thread.app = self

        self.marker_worker = TrajectoryDisplayWorker(
                                            frame_id="local_origin",
                                            marker_topic="trajectory_marker")
        self.marker_worker.moveToThread(self.ros_helper_thread)

        self.marker_path_worker = NodePathWorker(
                                            frame_id="local_origin",
                                            marker_topic="path_marker")
        self.marker_path_worker.moveToThread(self.ros_helper_thread)

        self.interactive_marker_worker = WaypointControlWorker(
                                            self.on_waypoint_control_callback,
                                            menu_callback=self.on_waypoint_menu_control_callback,
                                            frame_id="local_origin",
                                            marker_topic="trajectory_control")
        self.interactive_marker_worker.moveToThread(self.ros_helper_thread)

        self.animate_worker = AnimateWorker(self.on_animate_eval_callback,
                                            frame_id="minsnap_animation",
                                            parent_frame_id='local_origin',
                                            marker_topic="minsnap_animation_marker")
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

        self.send_trajectory_button.clicked.connect(self.on_send_trajectory_button_click)
        self.load_trajectory_button.clicked.connect(self.on_load_trajectory_button_click)
        self.save_trajectory_button.clicked.connect(self.on_save_trajectory_button_click)

        self.set_first_waypoint_to_drone_button.clicked.connect(self.on_set_first_waypoint_to_drone_button_click)
        self.set_first_waypoint_above_drone_button.clicked.connect(self.on_set_first_waypoint_above_drone_button_click)
        #self.close_loop_radio_button.clicked.connect(self.close_loop_radio_button_click)

        self.save_waypoints_from_traj_button.clicked.connect(self.on_save_waypoints_from_traj_click)

        self.add_node_button.clicked.connect(self.on_add_node_button_click)
        self.delete_node_button.clicked.connect(self.on_delete_node_button_click)

        self.plan_with_obstacles_button.clicked.connect(self.on_plan_with_obstacles_button_click)
        self.check_distance_button.clicked.connect(self.on_check_distance_button_click)
        self.show_freespace_checkbox.toggled.connect(self.on_show_freespace_button_click)

        double_validator = QDoubleValidator(parent=self.A_max_line_edit)
        double_validator.setBottom(0)
        self.A_max_line_edit.setValidator(double_validator)
        self.A_max_line_edit.setText(str(self.A_max))
        self.A_max_line_edit.textEdited.connect(self.on_A_max_line_edit_text_edit)

        self.A_max_select_line_edit.setValidator(double_validator)
        self.A_max_select_line_edit.setText(str(self.A_max))
        self.A_max_select_line_edit.textEdited.connect(self.on_A_max_select_line_edit_text_edit)

        self.terminal_text_edit.insertPlainText("Initialized freespace planner\nLoad an ESDF")
        self.path_dist_line_edit.setText("N/A")
        self.traj_dist_line_edit.setText("N/A")

        self.comp_time_line_edit.setValidator(double_validator)

        self.traj_time_line_edit.setValidator(double_validator)
        self.traj_accel_line_edit.setValidator(double_validator)

        double_validator_wayp = QDoubleValidator(parent=self.x_waypoint_line_edit)
        # rx = QRegExp("[+-]?[1-9]\\d+\\.\\d{2}")
        # double_validator_wayp = QRegExpValidator(rx,parent=self.y_waypoint_line_edit)
        # double_validator_wayp.setRange(-10.0, 100.0, 5)
        double_validator_wayp.setBottom(-100.0)
        double_validator_wayp.setTop(100.0)
        self.x_waypoint_line_edit.setValidator(double_validator_wayp)
        self.x_waypoint_line_edit.setText("0")
        self.x_waypoint_line_edit.textEdited.connect(self.on_x_waypoint_line_edit_text_edit)
        # double_validator_wayp = QDoubleValidator(parent=self.y_waypoint_line_edit)
        self.y_waypoint_line_edit.setValidator(double_validator_wayp)
        self.y_waypoint_line_edit.setText("0")
        self.y_waypoint_line_edit.textEdited.connect(self.on_y_waypoint_line_edit_text_edit)
        # double_validator_wayp = QDoubleValidator(parent=self.z_waypoint_line_edit)
        self.z_waypoint_line_edit.setValidator(double_validator_wayp)
        self.z_waypoint_line_edit.setText("0")
        self.z_waypoint_line_edit.textEdited.connect(self.on_z_waypoint_line_edit_text_edit)
        # double_validator_wayp = QDoubleValidator(parent=self.yaw_waypoint_line_edit)
        self.yaw_waypoint_line_edit.setValidator(double_validator_wayp)
        self.yaw_waypoint_line_edit.setText("0")
        self.yaw_waypoint_line_edit.textEdited.connect(self.on_yaw_waypoint_line_edit_text_edit)

        double_validator_i = QIntValidator(0,200,parent=self.index_line_edit)
        self.index_line_edit.setValidator(double_validator_i)
        self.index_line_edit.setText("0")
        self.index_line_edit.textEdited.connect(self.on_index_line_edit_text_edit)

        self.A_index_line_edit.setValidator(double_validator_i)
        self.A_index_line_edit.setText("0")
        self.A_index_line_edit.textEdited.connect(self.on_A_index_line_edit_text_edit)

        if ('disc_updated_signal' in self.global_dict.keys() and
                        self.global_dict['disc_updated_signal'] is not None):
            self.global_dict['disc_updated_signal'].connect(self.on_disc_updated_signal)

        if ('fsp_updated_signal_1' in self.global_dict.keys() and
                        self.global_dict['fsp_updated_signal_1'] is not None):
            self.global_dict['fsp_updated_signal_1'].connect(self.on_fsp_updated_signal)

        if ('fsp_updated_signal_2' in self.global_dict.keys() and
                        self.global_dict['fsp_updated_signal_2'] is not None):
            self.global_dict['fsp_updated_signal_2'].connect(self.on_fsp_updated_signal)

    def on_animate_radio_button_toggle(self, animate):
        print("In animate. flag is: {}".format(animate))
        if animate:
            print("start animation")
            self.animate_worker.start_publish()
            self.animate_worker.start_animate()
        else:
            print("stop animation")
            self.animate_worker.stop_publish()
            self.animate_worker.stop_animate()

    def on_animate_eval_callback(self, t):
        if self.qr_polytraj is None:
            return

        if not hasattr(self.qr_polytraj,"nodes"):
            print("Nodes not in qr_polytraj yet")
            return

        if self.qr_polytraj.nodes is None:
            # print("Nodes none in qr_polytraj")
            return

        if not self.qr_polytraj.restrict_freespace:
            print("REstrict freespace is false")
            return


        t_max = utils.seg_times_to_trans_times(self.qr_polytraj.times)[-1]
        if t > t_max:
            t = 0

        x = self.qr_polytraj.quad_traj['x'].piece_poly(t)
        y = self.qr_polytraj.quad_traj['y'].piece_poly(t)
        z = self.qr_polytraj.quad_traj['z'].piece_poly(t)
        yaw = self.qr_polytraj.quad_traj['yaw'].piece_poly(t)

        acc_vec = np.array([self.qr_polytraj.quad_traj['x'].piece_poly.derivative().derivative()(t),
                            self.qr_polytraj.quad_traj['y'].piece_poly.derivative().derivative()(t),
                            self.qr_polytraj.quad_traj['z'].piece_poly.derivative().derivative()(t)])

        q, data = body_frame.body_frame_from_yaw_and_accel( yaw, acc_vec, 'quaternion' )
        return (t_max, x, y, z, q[0], q[1], q[2], q[3])

    def on_fsp_updated_signal(self):
        print('in fsp callback')
        if self.global_dict['fsp_out_map'] is None:
            print("Spurious callback in taco; no esdf loaded")
            return

        if self.qr_polytraj is not None:
            # Setup as a freespace planner (will get nodes and sub waypoints)
            if self.qr_polytraj.nodes is None:
                self.setup_trajectory_for_freespace()


        # Update distances
        if hasattr(self,'l_max'):
            self.path_dist_line_edit.setText("{:.4f}".format(np.min(self.l_max)))
            # self.on_check_distance_button_click(check_trajectory=False)


    def on_A_max_line_edit_text_edit(self, text):
        try:
            self.A_max = float(text)*np.ones(np.shape(self.nodes['x'])[1]-1)
            self.A_max[0] = 3.0
            self.A_max[-1] = 3.0
        except ValueError as e:
            self.A_max_line_edit.text()
            self.A_max_line_edit.setText(str(self.A_max[0]))

    def on_A_max_select_line_edit_text_edit(self, text):
        try:
            A_select = float(text)
            self.A_max[self.A_sel_ind] = A_select
        except ValueError as e:
            self.A_max_select_line_edit.text()
            self.A_max_select_line_edit.setText("0")

    def on_A_index_line_edit_text_edit(self, text):
        try:
            self.A_sel_ind = int(text)
        except ValueError as e:
            self.A_index_line_edit.text()
            # self.index_line_edit.setText(str(self.index_waypoint))

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
    def on_index_line_edit_text_edit(self, text):
        try:
            self.index_waypoint = int(text)
        except ValueError as e:
            self.index_line_edit.text()
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

    def on_disc_updated_signal(self):
        if (self.global_dict['disc_out_waypoints'] is None or
                    'x' not in self.global_dict['disc_out_waypoints'].keys() or
                    'y' not in self.global_dict['disc_out_waypoints'].keys() or
                    'z' not in self.global_dict['disc_out_waypoints'].keys()):
            print("Spurious callback in minsnap; no discretized waypoints")
            return

        # self.nodes = self.global_dict['disc_out_waypoints']
        """ Initialise a default UNCO trajectory """
        waypoints = self.global_dict['disc_out_waypoints']
        costs = dict()
        costs['x'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['y'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['z'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['yaw'] = [0, 0, 1]  # minimum acceleration
        order=dict(x=9, y=9, z=9, yaw=5)
        waypoints = utils.form_waypoints_polytraj(waypoints,order)
        self.qr_polytraj = quadrotor_polytraj.QRPolyTraj(waypoints,
                                self.time_penalty,
                                costs=costs,
                                order=order,
                                yaw_eps = 1e-5)

        # Setup as a freespace planner (will get nodes and sub waypoints)
        self.setup_trajectory_for_freespace()

        # Display (but not trajectory)
        self.update_node_path_markers(self.n_samp_node_path)
        self.interactive_marker_worker.make_controls(self.nodes)
        self.interactive_marker_worker.update_controls(self.nodes)

        self.terminal_text_edit.appendPlainText("Nodes loaded.")

        # Update distances
        if hasattr(self,'l_max'):
            self.path_dist_line_edit.setText("{:.4f}".format(np.min(self.l_max)))
            # self.on_check_distance_button_click(check_trajectory=False)


    def on_check_distance_button_click(self):

        map = self.check_for_map()
        if map is None:
            return

        check_trajectory=True

        if not hasattr(self,'nodes'):
            print("Not checking: No nodes loaded")
            return
        if self.nodes is None:
            print('No node paths to check')
            return
        else:
            l_max = self.compute_nearest_distance_to_obstacles(self.nodes)
            print("Minimum distance from path to obstacles is {} m".format(np.min(l_max)))
            self.path_dist_line_edit.setText("{:.4f}".format(np.min(l_max)))

            if check_trajectory and self.qr_polytraj is not None:
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

    def on_plan_with_obstacles_button_click(self):
        map = self.check_for_map()
        if map is None:
            return

        self.controls_enabled = False
        self.plan_with_obstacles_button.setEnabled(False)

        # # Select different options for whether this is the first optimisation or not
        # if self.qr_polytraj.nodes is None:
        #     # First optimisation
        #
        #     #TODO(mereweth@jpl.nasa.gov) - push this to background thread
        #     # self.initialize_trajectory_object()
        #     self.setup_trajectory_for_freespace()

        # if np.sum(self.l_max<=0.0) >0:
        #     # Path in collision.
        #     return
            # else:
            #     self.terminal_text_edit.appendPlainText("Starting freespace planning.")
            #     print("Freespace planning finished")
            #     self.terminal_text_edit.appendPlainText("Freespace planning finished.")
            #     self.plan_with_obstacles_button.setEnabled(True)
            #     self.controls_enabled = True
            #     # Update GUI
            #     self.update_path_markers()
            #     self.update_node_path_markers()
            #     #self.interactive_marker_worker.make_controls(self.nodes)
            #     self.interactive_marker_worker.update_controls(self.nodes)
        # else:
        # # Check to see if path is in collision
        # l_max = self.compute_nearest_distance_to_obstacles(self.nodes)

        # Check if loaded trajectory is set up for freespace planning or not
        if not self.qr_polytraj.restrict_freespace:
            if self.qr_polytraj.nodes is None:
                # Setup trajectory if not already
                self.setup_trajectory_for_freespace()
            else:
                # Change flag if nodes already set
                self.qr_polytraj.restrict_freespace = True

        if np.sum(self.l_max<=0.0) > 0: # Assume here that l_max is computed every time a waypoint is modified
            out_msg = "Not forming waypoints as node path is in collision.\nFix and re-run"
            self.terminal_text_edit.appendPlainText(out_msg)
            print(out_msg)
            self.plan_with_obstacles_button.setEnabled(True)
            self.controls_enabled = True
            return

        # Subsequent optimisations
        self.terminal_text_edit.appendPlainText("Starting freespace planning.")

        # Set A_max from GUI
        print(self.A_max)
        print(self.qr_polytraj.A_max)
        A_max_new = self.A_max
        if True:#A_max_new[0] != self.qr_polytraj.A_max[0]:
            self.setup_trajectory_for_freespace()

        if np.sum(self.qr_polytraj.l_max<=0.0) > 0:
            out_msg = "Not planning as path is in collision\nReset and run again"
            self.terminal_text_edit.appendPlainText(out_msg)
            print(out_msg)
            self.plan_with_obstacles_button.setEnabled(True)
            self.controls_enabled = True
            return
        else:
            self.controls_enabled = False
            self.plan_with_obstacles_button.setEnabled(False)

            self.optimizer_thread = QThread()
            self.optimizer_thread.app = self

            self.optimizer_worker = OptimizerWorker(self.on_optimizer_worker_done)
            self.optimizer_worker.local_qr_polytraj = self.qr_polytraj
            self.optimizer_worker.moveToThread(self.optimizer_thread)
            self.optimizer_thread.started.connect(self.optimizer_worker.task)
            self.optimizer_thread.start()

    def on_optimizer_worker_done(self):
        self.qr_polytraj = self.optimizer_worker.local_qr_polytraj

        self.optimizer_worker.stop()
        self.optimizer_thread.quit()

        print("Freespace planning finished")
        self.terminal_text_edit.appendPlainText("Freespace planning finished.\nTotal time is: {:.3f} s\n".format(np.sum(self.qr_polytraj.times)))
        self.plan_with_obstacles_button.setEnabled(True)
        self.controls_enabled = True
        # Update GUI
        self.update_path_markers()
        self.interactive_marker_worker.update_controls(self.nodes)

        # Update distances
        self.on_check_distance_button_click()

    def initialize_trajectory_object(self):

        try:
            nodes = self.nodes
        except AttributeError:
            print("Error: Need to load in waypoints first")
            self.terminal_text_edit.appendPlainText("Need to load waypoints or a trajectory first.")
            return

        costs = dict()
        costs['x'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['y'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['z'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['yaw'] = [0, 0, 1]  # minimum acceleration
        order=dict(x=9, y=9, z=9, yaw=5)

        n_seg = nodes['x'].shape[1]-1
        num_internal = n_seg - 1
        der_fixed = utils.default_der_fixed_polytraj(num_internal,order)
        der_ineq = None

        # Setup der_fixed to be free at internal waypoints
        for key in der_fixed.keys():
            der_fixed[key][0,1:-1] = False

        # Store for later use
        self.der_fixed_nodes = der_fixed.copy()

        # Max Acceleraton
        if np.size(np.array(A_max)) == 1:
            self.A_max = np.ones(n_seg)*self.A_max


        l_max = self.compute_nearest_distance_to_obstacles(nodes)

        if sum(l_max<=0.0) > 0:
            out_msg = "No Trajectory planned as path is in collision!!!\nMove path to not be in collision"
            self.terminal_text_edit.appendPlainText(out_msg)
            print(out_msg)
            return

        self.qr_polytraj = quadrotor_polytraj.QRPolyTraj(nodes, self.time_penalty, costs=costs,
                                             order=order,der_fixed=der_fixed,
                                             restrict_freespace=True,
                                             l_max=l_max,A_max=self.A_max)

        self.terminal_text_edit.appendPlainText("Total trajectory time is: {:.3f} s\n".format(np.sum(self.qr_polytraj.times)))

    def on_waypoint_control_callback(self, position, index, qr_type="main"):
        if self.qr_polytraj.nodes is None:
            self.nodes['x'][0,index] = position['x']
            self.nodes['y'][0,index] = position['y']
            self.nodes['z'][0,index] = position['z']

            # Reset waypoints in qr_p if none currently exist
            self.qr_polytraj.waypoints = self.nodes.copy()

            self.update_node_path_markers(self.n_samp_node_path)
            # self.terminal_text_edit.appendPlainText("waypoint moved to {}".format(position))
            return

        if index > len(self.nodes['x'][0,:]):
            return

        if not self.controls_enabled:
            # TODO(mereweth@jpl.nasa.gov) - display error box
            # don't let user change controls
            self.interactive_marker_worker.update_controls(self.qr_polytraj.nodes,
                                                           indices=[index])
            return

        position["yaw"] = 0.0

        self.qr_polytraj.update_xyz_yaw_partial_nodes(index, position)
        # TODO(mereweth@jpl.nasa.gov) - return which waypoints/yaws changed by
        # more than some threshold?

        # Update changed nodes: computes the distance. Stored in l_max for the path
        if self.qr_polytraj.nodes is not None:
            self.update_changed_nodes()


        self.update_node_path_markers(self.n_samp_node_path)
        # Up to the user to know which controls they want to update
        # In this case, some yaws have changed but not been updated. This is
        # a performance tradeoff - see above
        self.interactive_marker_worker.update_controls(self.qr_polytraj.nodes,
                                                       indices=[index])

        # Update distances
        # self.on_check_distance_button_click(check_trajectory=False)
        self.path_dist_line_edit.setText("{:.4f}".format(np.min(self.l_max)))
        # self.terminal_text_edit.appendPlainText("waypoint moved to {}".format(position))

    def on_waypoint_menu_control_callback(self,command_type,data,qr_type="main"):

        if command_type == "delete":
            if self.nodes['x'].shape[1] == 2:
                print("Minimum number of waypoints is 2. Can not delete")
                return

            # Delete the waypoint
            index = data['index']

            # Delete waypoint
            self.delete_node(index)

            # Update markers
            self.update_node_path_markers(self.n_samp_node_path)
            self.interactive_marker_worker.make_controls(self.nodes)
            self.interactive_marker_worker.update_controls(self.nodes)

            # Update distances
            if hasattr(self,'l_max'):
                self.path_dist_line_edit.setText("{:.4f}".format(np.min(self.l_max)))
                # self.on_check_distance_button_click(check_trajectory=False)

        elif command_type == 'insert':
            # Insert the waypoint
            index = data['index']

            # Compute the position based on the neighbouring waypoints
            new_node = dict()
            for key in self.nodes.keys():
                if index == 0:
                    # New first node - extend out along same vector as between the previous first two waypoints
                    new_node[key] = self.nodes[key][0,index] + (self.nodes[key][0,index] - self.nodes[key][0,index+1])/2
                elif index == self.nodes[key].shape[1]:
                    # New last node
                    new_node[key] = self.nodes[key][0,index-1] + (self.nodes[key][0,index-1] - self.nodes[key][0,index-2])/2
                else:
                    new_node[key] = (self.nodes[key][0,index-1] + self.nodes[key][0,index])/2


            # Insert waypoint
            self.insert_node(new_node, index)

            # Update markers
            self.update_node_path_markers(self.n_samp_node_path)
            self.interactive_marker_worker.make_controls(self.nodes)
            self.interactive_marker_worker.update_controls(self.nodes)

            # Update distances
            if hasattr(self,'l_max'):
                self.path_dist_line_edit.setText("{:.4f}".format(np.min(self.l_max)))
                # self.on_check_distance_button_click(check_trajectory=False)

    def insert_node(self,new_node, index):
        """
            inserting a new node
        """

        if self.qr_polytraj.nodes is not None:
            self.qr_polytraj.insert_node(index, new_node)

            self.update_changed_nodes()

            self.A_max = self.qr_polytraj.A_max

        else:
            nodes_temp = dict()
            der_fixed_temp = dict()
            der_ineq_temp = dict()
            for key in new_node.keys():
                if np.size(new_node[key]) != self.nodes[key].shape[0]:
                    # Resize and fill with zeros if needed
                    insert_entry = np.concatenate([np.atleast_1d(new_node[key]),np.zeros(self.nodes[key].shape[0] - np.size(new_node[key]))])
                    insert_entry_der_fixed = np.zeros(self.nodes[key].shape[0],dtype=bool)
                else:
                    insert_entry = xyz_yaw_new_node[key]
                nodes_temp[key] = np.insert(self.nodes[key],index,insert_entry,axis=1)

                der_fixed_temp[key] = np.insert(self.der_fixed_nodes[key],index,insert_entry_der_fixed,axis=1)
                der_ineq_temp[key] = np.zeros(der_fixed_temp[key].shape,dtype=bool)

                if index == 0 : # If at the start: fix the start and free the second
                    # Fix the derivatives
                    der_fixed_temp[key][:,0] = True
                    der_fixed_temp[key][1:,1] = False
                    der_ineq_temp[key][:,0] = False
                elif index == self.nodes[key].shape[1]:# If at the end: fix the end and free the second last
                    # Fix the derivatives
                    der_fixed_temp[key][:,-1] = True
                    der_fixed_temp[key][1:,-2] = False
                    der_ineq_temp[key][:,-1] = False



            self.A_max = np.insert(self.A_max,index,self.A_max[np.min([index,np.shape(self.nodes['x'])[1]-1])])
            self.nodes = nodes_temp.copy()
            self.der_fixed_nodes = der_fixed_temp.copy()

            self.qr_polytraj.waypoints = nodes_temp.copy()
            self.qr_polytraj.der_fixed = der_fixed_temp.copy()
            self.qr_polytraj.der_ineq = der_ineq_temp.copy()


            l_max = self.compute_nearest_distance_to_obstacles(self.nodes)
            print("new l_max with insert node is {}".format(l_max))

    def delete_node(self, index):
        """
            Deleting a node
        """
        if self.qr_polytraj.nodes is not None:
            self.qr_polytraj.delete_node(index)

            self.update_changed_nodes()

            self.A_max = self.qr_polytraj.A_max
        else:
            nodes_temp = dict()
            der_fixed_temp = dict()
            der_ineq_temp = dict()

            for key in self.nodes.keys():
                nodes_temp[key] = np.delete(self.nodes[key],index,axis=1)
                der_fixed_temp[key] = np.delete(self.der_fixed_nodes[key],index,axis=1)
                der_ineq_temp[key] = np.zeros(der_fixed_temp[key].shape,dtype=bool)
                if index == 0:
                    # Fix the new starting point
                    der_fixed_temp[key][:,0] = True
                    der_ineq_temp[key][:,0] = False
                elif index == self.nodes[key].shape[1]-1:
                    # Fix the new end point
                    der_fixed_temp[key][:,-1] = True
                    der_ineq_temp[key][:,-1] = False

            self.nodes = nodes_temp.copy()
            self.qr_polytraj.waypoints = nodes_temp.copy()
            self.qr_polytraj.der_fixed = der_fixed_temp.copy()
            self.qr_polytraj.der_ineq = der_ineq_temp.copy()

            self.A_max = np.delete(self.A_max,np.min([index,np.shape(self.nodes['x'])[1]-1]))

            l_max = self.compute_nearest_distance_to_obstacles(self.nodes)

    def update_changed_nodes(self):
        """
            Reused functionality to create the waypoints from the nodes
        """

        print("in update_changed_nodes")
        self.nodes = self.qr_polytraj.nodes

        l_max = self.compute_nearest_distance_to_obstacles(self.qr_polytraj.nodes)

        if l_max is not None:
            # if self.qr_polytraj.A_max is None:
            self.qr_polytraj.A_max = self.A_max

            # Reset der_fixed
            for key in self.qr_polytraj.der_fixed.keys():
                self.qr_polytraj.der_fixed[key][0,1:-1] = False

            # Generate waypoints from the nodes
            self.qr_polytraj.waypoints_from_nodes(l_max,self.qr_polytraj.A_max)

            self.qr_polytraj.restrict_freespace = True


            print("WARNING: yaw not yet set from trajectory. Need to optimise to get yaw from trajectory")
            # for key in self.qr_polytraj.quad_traj.keys():
            #     self.qr_polytraj.quad_traj[key].get_piece_poly()
            #
            # #(TODO) Ideally get yaw from the waypoint
            # self.qr_polytraj.set_yaw_des_from_traj()

    def on_add_node_button_click(self):
        if self.nodes is None:
            print("No nodes received yet. Need to initialise")
            return

        if self.index_waypoint > self.nodes['x'].shape[1]:
            # Index too large
            print("Index is too large, not adding")
            return

        # Load from GUI
        new_node = dict()
        new_node['x'] = self.new_x_waypoint
        new_node['y'] = self.new_y_waypoint
        new_node['z'] = self.new_z_waypoint
        new_node['yaw'] = self.new_yaw_waypoint
        index = self.index_waypoint

        # Insert waypoint
        self.insert_node(new_node, index)

        # Update markers
        self.update_node_path_markers(self.n_samp_node_path)
        self.interactive_marker_worker.make_controls(self.nodes)
        self.interactive_marker_worker.update_controls(self.nodes)

        # Update distances
        if hasattr(self,'l_max'):
            self.path_dist_line_edit.setText("{:.4f}".format(np.min(self.l_max)))
            # self.on_check_distance_button_click(check_trajectory=False)

    def on_delete_node_button_click(self):
        if self.nodes is None:
            print("No Waypoints received yet. Need to initialise")
            return
        if self.index_waypoint >= self.nodes['x'].shape[1]:
            # Index too large
            print("Index is too large, waypoint does not exist")
            return
        if self.nodes['x'].shape[1] == 2:
            print("Minimum number of waypoints is 2. Can not delete")
            return

        # Load from GUI
        index = self.index_waypoint

        # Delete waypoint
        self.delete_node(index)

        # Update markers
        self.update_node_path_markers(self.n_samp_node_path)
        self.interactive_marker_worker.make_controls(self.nodes)
        self.interactive_marker_worker.update_controls(self.nodes)

        # Update distances
        if hasattr(self,'l_max'):
            self.path_dist_line_edit.setText("{:.4f}".format(np.min(self.l_max)))
            # self.on_check_distance_button_click(check_trajectory=False)

    def on_load_trajectory_button_click(self, checked=False, filename=None):
        map = self.check_for_map()
        if map is None:
            return

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
                self.yaw_eps = 1e-5
        except Exception as e:
            print("Could not load pickled QRPolyTraj from {}".format(filename))
            print(e.message)
            return


        self.update_path_markers()
        # Check if loaded trajectory is set up for freespace planning or not
        # Set up the trajectory for freespace planning
        self.setup_trajectory_for_freespace()

        # if not self.qr_polytraj.restrict_freespace:
        #     if self.qr_polytraj.nodes is None:
        #         self.setup_trajectory_for_freespace()
        #     else:
        #         self.qr_polytraj.restrict_freespace = True


        self.A_max_line_edit.setText(str(self.qr_polytraj.A_max[0]))

        self.update_node_path_markers(self.n_samp_node_path)
        self.interactive_marker_worker.make_controls(self.nodes)
        self.interactive_marker_worker.update_controls(self.nodes)

        # Update distances
        self.on_check_distance_button_click()
        self.terminal_text_edit.appendPlainText("Total trajectory time is: {:.3f} s\n".format(np.sum(self.qr_polytraj.times)))

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
        try:
            with open(filename, 'wb') as f:
                print("Saving pickled QRPolyTraj to {}".format(filename))
                pickle.dump(self.qr_polytraj, f, 2 )
        except Exception as e:
            print("Could not save pickled QRPolyTraj to {}".format(filename))
            print(e.message)

    def setup_trajectory_for_freespace(self):

        if self.qr_polytraj.nodes is None:
            # Take waypoints if nodes do note exist
            self.nodes = self.qr_polytraj.waypoints.copy()
            self.der_fixed_nodes = self.qr_polytraj.der_fixed.copy()
            if np.size(np.array(self.A_max)) <= 1:
                self.A_max = np.ones(np.shape(self.nodes['x'])[1])*self.A_max
        else:
            self.nodes = self.qr_polytraj.nodes
            if np.size(np.array(self.A_max)) <= 1:
                self.A_max = np.ones(np.shape(self.nodes['x'])[1])*self.A_max

        # Take a trajectory not setup for freespace planning and set it up for that
        l_max = self.compute_nearest_distance_to_obstacles(self.nodes)
        # print(l_max)

        if np.sum(l_max<=0.0) > 0:
            out_msg = "Not forming waypoints as node path is in collision.\nFix and re-run"
            self.terminal_text_edit.appendPlainText(out_msg)
            print(out_msg)
            return

        if l_max is not None:
            # if self.qr_polytraj.A_max is None:
            self.qr_polytraj.A_max = self.A_max

            # Reset der_fixed
            for key in self.qr_polytraj.der_fixed.keys():
                self.qr_polytraj.der_fixed[key][0,1:-1] = False

            # Generate waypoints from the nodes
            self.qr_polytraj.waypoints_from_nodes(l_max,self.qr_polytraj.A_max)

            self.qr_polytraj.restrict_freespace = True

    def on_set_first_waypoint_to_drone_button_click(self):

        if self.last_orientation is None or self.last_position is None:
            print("not setting first waypoint; none received yet")
            return

        try:
            q = [self.last_orientation['w'],
                 self.last_orientation['x'],
                 self.last_orientation['y'],
                 self.last_orientation['z']]

            position = self.last_position

        except AttributeError:
            print("not setting first waypoint; none received yet")
            return
        out_msg = "Updating the first and last waypoint... Position: {}\nOrientation: {}".format(position,self.last_orientation)
        print(out_msg)
        # self.terminal_text_edit.appendPlainText(out_msg)


        yaw = tf.transformations.euler_from_quaternion(q)[2]

        self.update_first_waypoint( position, yaw, defer=True)
        self.update_last_waypoint( position, yaw, defer=False)
        self.t=0 #reset visualization

        if self.qr_polytraj is not None:
            self.update_path_markers()
        self.update_node_path_markers(self.n_samp_node_path)
        #self.interactive_marker_worker.make_controls(self.nodes)
        self.interactive_marker_worker.update_controls(self.nodes)

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
        out_msg="Updating the first and last waypoint... Position: {}\nOrientation: {}".format(position,self.last_orientation)
        print(out_msg)
        # self.terminal_text_edit.appendPlainText(out_msg)

        yaw = tf.transformations.euler_from_quaternion(q)[2]

        self.update_first_waypoint( position, yaw, defer=True)
        position['z'] += 1.0
        self.update_last_waypoint( position, yaw, defer=False)
        self.t=0 #reset visualization

        if self.qr_polytraj is not None:
            self.update_path_markers()
        self.update_node_path_markers(self.n_samp_node_path)
        #self.interactive_marker_worker.make_controls(self.nodes)
        self.interactive_marker_worker.update_controls(self.nodes)

    def update_waypoint(self,position, yaw, index):
        xyz_yaw_partial_waypoint = dict()

        for key in position.keys():
            xyz_yaw_partial_waypoint[key] = position[key]
        xyz_yaw_partial_waypoint["yaw"] = yaw

        if self.qr_polytraj is not None:
            self.qr_polytraj.update_xyz_yaw_partial_nodes(index, xyz_yaw_partial_waypoint)
            self.nodes = self.qr_polytraj.nodes

            self.qr_polytraj.do_update_trajectory_markers = True
            self.qr_polytraj.do_update_control_yaws = True
        else:
            for key in position.keys():
                self.nodes[key][0,index] = position[key]
            self.nodes['yaw'][0,index] = yaw

        # self.distance_current = False

    def update_first_waypoint(self,position, yaw, defer=False):
        position['yaw'] = yaw
        self.on_waypoint_control_callback(position, 0)
        # self.update_waypoint(position, yaw, 0)

    def update_last_waypoint(self,position, yaw, defer=False):
        print(position)
        position['yaw'] = yaw
        self.on_waypoint_control_callback(position, self.nodes['x'].shape[1]-1)
        # self.update_waypoint(position, yaw, self.nodes['x'].shape[1]-1)

    def compute_nearest_distance_to_obstacles(self,nodes):

        map = self.check_for_map()
        if map is None:
            return

        n_seg = nodes['x'].shape[1] - 1
        n_samples = 1000 # TODO (bmorrell@jpl.nasa.gov) Make this a setting in qr_polytraj
        l_max = np.zeros(n_seg)

        for k in range(0,n_seg):
            # Create query points
            query = np.matrix(np.zeros([3,n_samples]),dtype='double')
            dist = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='double')
            obs = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='int32')

            # load in x, y, z points
            query[0,:] = np.around(np.linspace(nodes['x'][0,k],nodes['x'][0,k+1],n_samples),4)
            query[1,:] = np.around(np.linspace(nodes['y'][0,k],nodes['y'][0,k+1],n_samples),4)
            query[2,:] = np.around(np.linspace(nodes['z'][0,k],nodes['z'][0,k+1],n_samples),4)

            # Query the database
            map.getDistanceAtPosition(query, dist, obs)

            if self.unknown_is_free:
                dist[obs != 1] = 2.0

            # Find the minimum obstacle distance
            l_max[k] = np.min(dist) - self.quad_buffer

        # check for collisions
        if np.min(l_max <=0.0):
            out_msg = "Path is in collision with environment"
            print(out_msg)
            self.terminal_text_edit.appendPlainText(out_msg)

        # Update l_max in the qr_polytraj object
        # print("setting l_max")
        self.l_max = l_max
        return l_max

    def check_distance_to_trajectory(self,x=None,y=None,z=None,samp_mult=100):
        map = self.check_for_map()
        if map is None:
            return

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

    def update_path_markers(self):
        if self.qr_polytraj is None:
            print("Not publishing markers as self.qr_polytraj is None")
            return

        trans_times = utils.seg_times_to_trans_times(self.qr_polytraj.times)
        t_total = trans_times[-1] - trans_times[0]
        t = np.linspace(trans_times[0], trans_times[-1], t_total*100)
        x = self.qr_polytraj.quad_traj['x'].piece_poly(t)
        y = self.qr_polytraj.quad_traj['y'].piece_poly(t)
        z = self.qr_polytraj.quad_traj['z'].piece_poly(t)
        print("Updating Path")

        if self.plot_traj_color:
            if not hasattr(self,'dist_traj'):
                dist = self.check_distance_to_trajectory(x,y,z)
            elif len(self.dist_traj) != len(x):
                dist = self.check_distance_to_trajectory(x,y,z)
            else:
                dist = self.dist_traj
        else:
            dist = None

        self.marker_worker.publish_path(x, y, z, dist)

        t = np.linspace(trans_times[0], trans_times[-1], t_total*10)
        x = self.qr_polytraj.quad_traj['x'].piece_poly(t)
        y = self.qr_polytraj.quad_traj['y'].piece_poly(t)
        z = self.qr_polytraj.quad_traj['z'].piece_poly(t)
        ax = self.qr_polytraj.quad_traj['x'].piece_poly.derivative().derivative()(t)
        ay = self.qr_polytraj.quad_traj['y'].piece_poly.derivative().derivative()(t)
        az = self.qr_polytraj.quad_traj['z'].piece_poly.derivative().derivative()(t)

        self.marker_worker.publish_quiver(x, y, z, ax, ay, az)

        max_a = np.amax(np.sqrt(ax**2+ay**2+az**2))

        self.update_results_display(max_a)

    def update_node_path_markers(self, n_steps):

        # Generate the path
        nodes = self.nodes
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

        if hasattr(self.qr_polytraj,'waypoints'):
            self.marker_path_worker.publish_sub_waypoints(self.qr_polytraj.waypoints)
        # if hasattr(self,'l_max'):
        #     self.marker_path_worker.publish_tubes(self.nodes,self.l_max)

    def on_send_trajectory_button_click(self):
        if self.qr_polytraj is None:
            return
            self.terminal_text_edit.appendPlainText("Nothing to send")
        msg = PolyTraj_msg()
        self.qr_polytraj.fill_poly_traj_msg(msg)
        self.polytraj_pub.publish(msg)
        self.terminal_text_edit.appendPlainText("Trajectory sent.")

    def on_show_freespace_button_click(self, show_freespace):
        if self.nodes is not None:
            # Disable Button
            if show_freespace:#self.show_freespace_radio_button.isChecked():
                # if hasattr(self,'l_max'):
                #     self.marker_path_worker.publish_tubes(self.nodes,self.l_max)
                # else:
                # TODO (bmorrell@jpl.nasa.gov) Have a check for whether or not the distances need to be updated
                self.compute_nearest_distance_to_obstacles(self.nodes)
                self.marker_path_worker.publish_tubes(self.nodes,self.l_max)
            else:
                # Clear markers
                self.marker_path_worker.hide_tubes(self.nodes)

    def close_loop_radio_button_click(self):
        if self.nodes is not None:
            # Disable Button
            if self.close_loop_radio_button.isChecked():
                # if hasattr(self,'l_max'):
                #     self.marker_path_worker.publish_tubes(self.nodes,self.l_max)
                # else:
                # TODO (bmorrell@jpl.nasa.gov) Have a check for whether or not the distances need to be updated
                self.compute_nearest_distance_to_obstacles(self.nodes)
                self.marker_path_worker.publish_tubes(self.nodes,self.l_max)
            else:
                # Clear markers
                self.marker_path_worker.hide_tubes(self.nodes)


    def check_for_map(self):
        try:
            map = self.global_dict['fsp_out_map']
        except AttributeError:
            out_msg = "Not computing nearest distance. Need to have loaded an ESDF"
            print(out_msg)
            self.terminal_text_edit.appendPlainText(out_msg)
            return None

        if map is None:
            out_msg = "Not computing nearest distance. Need to have loaded an ESDF"
            print(out_msg)
            self.terminal_text_edit.appendPlainText(out_msg)
            return None

        return map

    def update_results_display(self, max_a=-1.0):
        """
        Set text for comp time and iterations to display in the GUI
        """
        comp_time = 0.0
        for key in self.qr_polytraj.quad_traj.keys():
            comp_time += self.qr_polytraj.quad_traj[key].opt_time
        self.comp_time_line_edit.setText(str(comp_time))
        # self.comp_time_line_edit.setText(str(self.qr_polytraj.opt_time))
        self.traj_time_line_edit.setText(str(np.sum(self.qr_polytraj.times)))
        self.traj_accel_line_edit.setText(str(max_a))
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

def main():
    from torq_gcs.plan.taco import FreespacePlannerGUI

    app = QApplication( sys.argv )

    global_dict = dict()
    fs_planner = FreespacePlannerGUI(global_dict, new_node=True)

    try:
        import imp
        conf = imp.load_source('torq_config',
                        os.path.abspath('~/Desktop/environments/344.py'))
        conf.torq_config(fs_planner)
    except Exception as e:
        print("Error in config script")
        print(e)

    fs_planner.show()

    return app.exec_()

if __name__=="__main__":
    sys.exit(main())
