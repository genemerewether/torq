#!/usr/bin/env python

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

import rospy, rospkg
from geometry_msgs.msg import Point
from visualization_msgs.msg import *
from interactive_markers.menu_handler import *
from interactive_markers.interactive_marker_server import *
import tf
import numpy as np
from python_qt_binding.QtCore import QObject, Signal, QTimer
from diffeo import body_frame

import transforms3d
from transforms3d import quaternions

import re

TRAJ_PATH_ID  = 2
NODE_PATH_ID  = 5
TUBE_ID       = 10
SPHERE_ID     = 200
SUB_WAY_ID    = 100

TRAJ_ACCEL_ID = 30000  #one for each arrow

def build_quad_marker_template():
    quad_marker = Marker()
    quad_marker.type = quad_marker.MESH_RESOURCE
    quad_marker.scale.x = 0.001
    quad_marker.scale.y = 0.001
    quad_marker.scale.z = 0.001
    quad_marker.color.r = 0.5
    quad_marker.color.g = 0.5
    quad_marker.color.b = 1
    quad_marker.color.a = 1
    quad_marker.mesh_resource = 'package://torq_gcs/mesh/TORQ_250_041917_17k.ply'
    quad_marker.mesh_use_embedded_materials = True
    return quad_marker



# This class will need to be duplicated & renamed when different functionality
# is needed for different planners
class TrajectoryDisplayWorker(QObject):
    finished = Signal() # class variable shared by all instances

    def __init__(self, finished_callback=None,
                 node_name='trajectory_display_worker',
                 frame_id='local_origin',
                 marker_topic='trajectory_marker',
                 new_node=False,
                 parent=None,
                 qr_type="main"):
        super(TrajectoryDisplayWorker, self).__init__(parent)

        if new_node:
            rospy.init_node(node_name, anonymous=True)

        if finished_callback is not None:
            self.finished.connect(finished_callback)

        #TODO(mereweth@jpl.nasa.gov) - how to shut down?
        self.is_running = True
        self.qr_type = qr_type

        self.frame_id = frame_id
        self.marker_pub = rospy.Publisher(marker_topic, MarkerArray,
                                          queue_size=100)

    def publish_path(self, x, y, z,dist=None,col_limit = 1.0):
        path_marker = Marker()
        path_marker.header.frame_id = self.frame_id
        path_marker.type = path_marker.LINE_STRIP
        if self.qr_type is "main":
            path_marker.id = TRAJ_PATH_ID
        elif self.qr_type is "entry":
            path_marker.id = TRAJ_PATH_ID+1
        elif self.qr_type is "exit":
            path_marker.id = TRAJ_PATH_ID+2
        path_marker.action = path_marker.ADD
        path_marker.scale.x = 0.03 #line width
        path_marker.pose.orientation.w = 1.0
        path_marker.pose.position.x = 0
        path_marker.pose.position.y = 0
        path_marker.pose.position.z = 0

        if dist is None:
            if self.qr_type is "main":
                path_marker.color.a = 1.0
                path_marker.color.r = 1.0
                path_marker.color.g = 0.5
                path_marker.color.b = 0.5
            elif self.qr_type is "entry":
                path_marker.color.a = 1.0
                path_marker.color.r = 0.5
                path_marker.color.g = 1.0
                path_marker.color.b = 0.5
            elif self.qr_type is "exit":
                path_marker.color.a = 1.0
                path_marker.color.r = 1.0
                path_marker.color.g = 0.0
                path_marker.color.b = 0.0

        for i in range(0,len(x)):
            path_marker.points.append(Point(x[i], y[i], z[i]))
            if dist is not None:
                if dist[i] <= 0:
                    # Red
                    c = std_msgs.msg.ColorRGBA(1.0,0,0,1.0)
                elif dist[i] > col_limit:
                    # Green
                    c = std_msgs.msg.ColorRGBA(0.0,1.0,0,1.0)
                else:
                    # scale color
                    g = dist[i]/col_limit
                    r = 1-g
                    c = std_msgs.msg.ColorRGBA(r,g,0,1.0)

                path_marker.colors.append(c)

        marker_array = MarkerArray()
        marker_array.markers.append( path_marker )
        self.marker_pub.publish(marker_array)

    def publish_quiver(self, x, y, z, ax, ay, az):
        quiver_marker_array = []
        for i in range(0,len(x)):
            quiver_marker = Marker()
            quiver_marker.header.frame_id = self.frame_id
            quiver_marker.type = quiver_marker.ARROW
            quiver_marker.id = TRAJ_ACCEL_ID+i
            quiver_marker.action = quiver_marker.ADD
            quiver_marker.scale.x = 0.01 #line width
            quiver_marker.scale.y = 0.02 #line width
            quiver_marker.color.a = 1.0
            quiver_marker.color.r = 1.0 if i%10==0 else 0.2
            quiver_marker.color.g = 0.2 if i%10==0 else 1.0
            quiver_marker.color.b = 0.2 if i%10==0 else 0.2
            scale = 0.05
            quiver_marker.points.append(Point(x[i], y[i], z[i]))
            quiver_marker.points.append(Point(x[i] + scale * ax[i],
                                              y[i] + scale * ay[i],
                                              z[i] + scale * az[i]))# + scale*9.81))
            quiver_marker_array.append(quiver_marker)
        for i in range(len(x),len(x)+1000):
            quiver_marker = Marker()
            quiver_marker.header.frame_id = self.frame_id
            quiver_marker.type = quiver_marker.ARROW
            quiver_marker.id = TRAJ_ACCEL_ID+i
            quiver_marker.action = quiver_marker.DELETE
            quiver_marker_array.append(quiver_marker)

        marker_array = MarkerArray()
        marker_array.markers += quiver_marker_array
        self.marker_pub.publish(marker_array)

    def publish_marker_spheres(self,turning_points):
        # spheres showing collision locations

        marker_array = MarkerArray()

        for i in range(1,turning_points['x'].size):
            sphere_marker = Marker()
            sphere_marker.header.frame_id = self.frame_id
            sphere_marker.type = sphere_marker.SPHERE
            sphere_marker.id = SPHERE_ID+i
            sphere_marker.action = sphere_marker.ADD
            # Scale
            radius = 0.3
            sphere_marker.scale.x = radius
            sphere_marker.scale.y = radius
            sphere_marker.scale.z = radius
            sphere_marker.pose.orientation.w = 1.0
            sphere_marker.pose.position.x = 0
            sphere_marker.pose.position.y = 0
            sphere_marker.pose.position.z = 0

            # Color
            sphere_marker.color.r = 1.0
            sphere_marker.color.g = 0.0
            sphere_marker.color.b = 0.0
            sphere_marker.color.a = 0.45

            # Position
            sphere_marker.pose.position.x = turning_points['x'][i]
            sphere_marker.pose.position.y = turning_points['y'][i]
            sphere_marker.pose.position.z = turning_points['z'][i]

            marker_array.markers.append( sphere_marker )
        self.marker_pub.publish(marker_array)

    def hide_marker_spheres(self):
        marker_array = MarkerArray()

        for i in range(30000):
            sphere_marker = Marker()
            sphere_marker.header.frame_id = self.frame_id
            sphere_marker.type = sphere_marker.SPHERE
            sphere_marker.id = SPHERE_ID+i
            sphere_marker.action = sphere_marker.DELETE

            marker_array.markers.append( sphere_marker )
        self.marker_pub.publish(marker_array)

    def stop(self):
        self.is_running = False

class NodePathWorker(QObject):
    finished = Signal() # class variable shared by all instances

    def __init__(self, finished_callback=None,
                 node_name='path_display_worker',
                 frame_id='local_origin',
                 marker_topic='path_marker',
                 new_node=False,
                 parent=None):
        super(NodePathWorker, self).__init__(parent)

        if new_node:
            rospy.init_node(node_name, anonymous=True)

        if finished_callback is not None:
            self.finished.connect(finished_callback)

        #TODO(mereweth@jpl.nasa.gov) - how to shut down?
        self.is_running = True

        self.frame_id = frame_id
        self.marker_pub = rospy.Publisher(marker_topic, MarkerArray,
                                          queue_size=100)

    def publish_node_path(self, x, y, z,dist=None,col_limit = 1.0):
        path_marker = Marker()
        path_marker.header.frame_id = self.frame_id
        path_marker.type = path_marker.LINE_STRIP
        path_marker.id = NODE_PATH_ID
        path_marker.action = path_marker.ADD
        path_marker.scale.x = 0.015 #line width
        path_marker.pose.orientation.w = 1.0
        path_marker.pose.position.x = 0
        path_marker.pose.position.y = 0
        path_marker.pose.position.z = 0

        alpha = 0.8

        if dist is None:
            path_marker.color.a = alpha
            path_marker.color.r = 0.5
            path_marker.color.g = 0.5
            path_marker.color.b = 0.5

        for i in range(0,len(x)):
            path_marker.points.append(Point(x[i], y[i], z[i]))
            if dist is not None:
                if dist[i] <= 0:
                    # Black
                    c = std_msgs.msg.ColorRGBA(1.0,0.,0,alpha)
                elif dist[i] > col_limit:
                    # White
                    c = std_msgs.msg.ColorRGBA(0.,0.0,1.0,alpha)
                else:
                    # scale color
                    g = dist[i]/col_limit
                    c = std_msgs.msg.ColorRGBA(g,g,g,alpha)

                path_marker.colors.append(c)

        marker_array = MarkerArray()
        marker_array.markers.append( path_marker )
        self.marker_pub.publish(marker_array)

    def publish_sub_waypoints(self,waypoints):
        # Low level waypoints - plot as spheres
        waypoint_marker = Marker()

        waypoint_marker.header.frame_id = self.frame_id
        waypoint_marker.type = waypoint_marker.SPHERE_LIST
        waypoint_marker.id = SUB_WAY_ID
        waypoint_marker.action = waypoint_marker.ADD
        radius = 0.05
        waypoint_marker.scale.x = radius #radius
        waypoint_marker.scale.y = radius #radius
        waypoint_marker.scale.z = radius #radius
        waypoint_marker.pose.orientation.w = 1.0
        waypoint_marker.pose.position.x = 0
        waypoint_marker.pose.position.y = 0
        waypoint_marker.pose.position.z = 0
        waypoint_marker.color.r = 0.0
        waypoint_marker.color.g = 1.0
        waypoint_marker.color.b = 1.0
        waypoint_marker.color.a = 1.0

        for i in range(0, len(waypoints['x'][0,:]) ):
            waypoint_marker.points.append(Point(waypoints['x'][0,i],waypoints['y'][0,i],waypoints['z'][0,i]))

        marker_array = MarkerArray()
        marker_array.markers.append( waypoint_marker )
        self.marker_pub.publish(marker_array)

    def publish_tubes(self,nodes,l_max):
        # TUbes showing freespace for TACO

        marker_array = MarkerArray()

        for i in range(1,nodes['x'].shape[1]):
            tube_marker = Marker()
            tube_marker.header.frame_id = self.frame_id
            tube_marker.type = tube_marker.CYLINDER
            tube_marker.id = TUBE_ID+i
            tube_marker.action = tube_marker.ADD
            r = l_max[i-1]

            p0 = np.array([nodes['x'][0,i-1],nodes['y'][0,i-1],nodes['z'][0,i-1]])
            p1 = np.array([nodes['x'][0,i],nodes['y'][0,i],nodes['z'][0,i]])

            tube_marker = self.make_cylinder(tube_marker,p0,p1,r)

            marker_array.markers.append( tube_marker )
        self.marker_pub.publish(marker_array)

    def make_cylinder(self,tube_marker,p0,p1,r):
        # Cylinder size
        # X and y as the diameter
        tube_marker.scale.x = 2*r
        tube_marker.scale.y = 2*r

        # compute the orientation and length
        # Vector between adjacent nodes
        axis_vec = p1 - p0

        # mid point
        mid_point = p0 + axis_vec/2

        # Length of vector
        length = np.linalg.norm(axis_vec)
        axis_vec /= length

        # Angle between vector and z axis
        cos_ang = axis_vec.dot(np.array([0.0,0.0,1.]))

        # Perpindicular vector
        rot_axis = np.cross(np.array([0.0,0.0,1.]),axis_vec)
        rot_length = np.linalg.norm(rot_axis)
        rot_axis /= rot_length

        # Use atan2 to get the angle
        angle = np.arctan2(rot_length,cos_ang)

        # Set the length
        tube_marker.scale.z = length + 2*r
        # Set the orientaiton
        tube_marker.pose.orientation.w = np.cos(angle/2)
        tube_marker.pose.orientation.x = rot_axis[0]*np.sin(angle/2) # Without making it a normal vector it is already multiplied by sin(theta)
        tube_marker.pose.orientation.y = rot_axis[1]*np.sin(angle/2)
        tube_marker.pose.orientation.z = rot_axis[2]*np.sin(angle/2)
        # Set the position
        tube_marker.pose.position.x = mid_point[0]
        tube_marker.pose.position.y = mid_point[1]
        tube_marker.pose.position.z = mid_point[2]

        tube_marker.color.a = 0.2
        tube_marker.color.r = 1.0
        tube_marker.color.g = 0.0
        tube_marker.color.b = 0.0

        return tube_marker

    def hide_tubes(self,nodes):
        marker_array = MarkerArray()

        for i in range(1,nodes['x'].shape[1]):
            tube_marker = Marker()
            tube_marker.header.frame_id = self.frame_id
            tube_marker.type = tube_marker.CYLINDER
            tube_marker.id = TUBE_ID+i
            tube_marker.action = tube_marker.DELETE

            marker_array.markers.append( tube_marker )
        self.marker_pub.publish(marker_array)


    def stop(self):
        self.is_running = False

# This class will need to be duplicated & renamed when different functionality
# is needed for different planners
class WaypointControlWorker(QObject):
    finished = Signal() # class variable shared by all instances

    def __init__(self,
                 control_callback,
                 menu_callback=None,
                 finished_callback=None,
                 node_name='waypoint_control_worker',
                 frame_id='local_origin',
                 marker_topic='trajectory_control',
                 new_node=False,
                 parent=None,
                 use_menu=True,
                 qr_type="main",
                 entry_ID=0,
                 exit_ID=0):
        super(WaypointControlWorker, self).__init__(parent)

        if new_node:
            rospy.init_node(node_name, anonymous=True)

        if finished_callback is not None:
            self.finished.connect(finished_callback)

        #TODO(mereweth@jpl.nasa.gov) - how to shut down?
        self.is_running = True

        self.use_menu = use_menu
        self.qr_type = qr_type

        self.entry_ID = entry_ID
        self.exit_ID = exit_ID

        self.control_callback = control_callback
        self.menu_callback = menu_callback

        self.frame_id = frame_id

        self.marker_server = InteractiveMarkerServer(marker_topic)

        # if self.use_menu:
        #     self.init_menu()

    # TODO(mereweth@jpl.nasa.gov) - how to maximize reusability?
    # what waypoint format - full pose? or x,y,z,yaw?
    def make_controls(self, waypoints, closed_loop=False):
        self.marker_server.clear()
        self.marker_server.applyChanges()

        # Set tracking of interactive waypoint states
        self.check_tracking = np.zeros(len(waypoints['x'][0,:]))
        self.check_tracking[self.entry_ID] += 1
        self.check_tracking[self.exit_ID] += 2

        # Create menus for each waypoint
        self.init_menu(waypoints)

        print("Creating " + str(len(waypoints['x'][0,:])) + " trajectory controls")

        for i in range(0, len(waypoints['x'][0,:]) ):
            if closed_loop and i == (len(waypoints['x'][0,:])-1):
                print("ignoring last waypoint")
                continue
                # Do not create last waypoint control
            if self.qr_type=='entry' and i == (len(waypoints['x'][0,:])-1):
                # Don't move the waypoint that is fixed to the trajectory
                continue
            if self.qr_type=='exit' and i == 0:
                # Don't move the waypoint that is fixed to the trajectory
                continue

            # create an interactive marker
            int_marker = InteractiveMarker()
            int_marker.header.frame_id = self.frame_id
            int_marker.scale = 0.2
            int_marker.pose.position.x = waypoints['x'][0,i]
            int_marker.pose.position.y = waypoints['y'][0,i]
            int_marker.pose.position.z = waypoints['z'][0, i]
            # q = tf.transformations.quaternion_from_euler(1.5,0,waypoints['yaw'][0,i])
            # int_marker.pose.orientation.x = q[0]
            # int_marker.pose.orientation.y = q[1]
            # int_marker.pose.orientation.z = q[2]
            # int_marker.pose.orientation.w = q[3]
            int_marker.name = str(i)+self.qr_type
            # create a non-interactive control which contains the box
            quad_control = InteractiveMarkerControl()
            quad_control.always_visible = True
            quad_control.orientation_mode = InteractiveMarkerControl.VIEW_FACING
            quad_control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
            quad_control.independent_marker_orientation = True

            quad_marker = build_quad_marker_template()
            if self.qr_type is "main" and i!=self.entry_ID and i!=self.exit_ID:
                quad_marker.color.r = 1.0
                quad_marker.color.g = 0.5
                quad_marker.color.b = 0.5
            elif self.qr_type is "entry" or (i == self.entry_ID and self.qr_type is "main"):
                quad_marker.color.r = 0.5
                quad_marker.color.g = 1.0
                quad_marker.color.b = 0.5
            elif self.qr_type is "exit" or (i == self.exit_ID and self.qr_type is "main"):
                quad_marker.color.r = 1.0
                quad_marker.color.g = 0.0
                quad_marker.color.b = 0.0

            quad_control.markers.append( quad_marker )
            int_marker.controls.append( quad_control )

            for ox,oy,oz,name in [(1,0,0,'move_x'),(0,1,0,'move_z'),(0,0,1,'move_y'),(0,1,0,'rotate_z')]:
                #not sure why z and y are swapped
                control = InteractiveMarkerControl()
                control.orientation.w = 1
                control.orientation.x = ox
                control.orientation.y = oy
                control.orientation.z = oz
                control.name = name
                if name[0]=='m':
                    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
                else:
                    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
                int_marker.controls.append(control)

            if self.use_menu:
                # Initialise the controls
                menu_control = InteractiveMarkerControl()

                # Set the interaction mode
                menu_control.interaction_mode = InteractiveMarkerControl.BUTTON
                menu_control.always_visible = True
                menu_control.name = 'menu'

                # Append a marker to the control
                menu_control.markers.append(quad_marker)

                # Append the controls to the interactive marker
                int_marker.controls.append(menu_control)

            # callback probably needs to be in this thread for ROS
            # Then, that callback can use Qt signals and slots
            self.marker_server.insert( int_marker, self.waypoint_control_callback )


            if self.use_menu:
                self.menu_handler[str(i)].apply( self.marker_server, int_marker.name)
                # if True:#i==0:
                #     # Call again to ativate both menu and move-interactivity for the first waypoint
                #     self.marker_server.insert( int_marker, self.waypoint_control_callback )
                #     self.menu_handler[key].apply( self.marker_server, int_marker.name)

        self.marker_server.applyChanges()

    # TODO(mereweth@jpl.nasa.gov) - how to maximize reusability?
    # what waypoint format - full pose? or x,y,z,yaw?
    # who is reponsible for recording a change in type of marker?
    def update_controls(self, waypoints, indices=None, closed_loop=False, acc_wp = None):

        n_waypoints = len(waypoints['x'][0,:])

        if indices is None:
            indices = range(0, n_waypoints)

        if closed_loop:
            # Do not create a marker for the last waypoint
            indices = np.delete(indices,np.where(indices==n_waypoints-1))

        for i in indices:
            if closed_loop and i == (len(waypoints['x'][0,:])-1):
                print("ignoring last waypoint")
                continue
                # Do not create last waypoint control
            if self.qr_type=='entry' and i == (len(waypoints['x'][0,:])-1):
                # Don't move the waypoint that is fixed to the trajectory
                continue
            if self.qr_type=='exit' and i == 0:
                # Don't move the waypoint that is fixed to the trajectory
                continue
            pose = geometry_msgs.msg.Pose()
            pose.position.x = waypoints['x'][0,i]
            pose.position.y = waypoints['y'][0,i]
            pose.position.z = waypoints['z'][0,i]
            # if acc_wp is None:
            #     q = tf.transformations.quaternion_from_euler(0.0,0.0,waypoints['yaw'][0,i])
            #     pose.orientation.w = q[3]
            #     pose.orientation.x = q[0]
            #     pose.orientation.y = q[1]
            #     pose.orientation.z = q[2]
            # else:
            # q, data = body_frame.body_frame_from_yaw_and_accel( waypoints['yaw'][0,i], acc_wp[:,i], out_format='quaternion' ,deriv_type='yaw_only')
            # HACK QUICK FIX. Even cleaner would be to just set to a unit quaternion
            # q = tf.transformations.quaternion_from_euler(0.0,0.0,waypoints['yaw'][0,i])
            # pose.orientation.w = q[0]
            # pose.orientation.x = q[1]
            # pose.orientation.y = q[2]
            # pose.orientation.z = q[3]
            
            pose.orientation.w = 1.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            self.marker_server.setPose( str(i)+self.qr_type, pose )

            self.marker_server.applyChanges()

    def index_from_name(self, name):
        return int(re.search(r'\d+', name).group())

    def waypoint_control_callback(self, callback):
        #compute difference vector for this cube
        position = dict()
        position['x'] = callback.pose.position.x
        position['y'] = callback.pose.position.y
        position['z'] = callback.pose.position.z
        index = self.index_from_name(callback.marker_name)

        #if index > len(qr_p.waypoints['x'][0,:]):
        #    return

        if callback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            print("Waypoint control callback for index " + str(index))

            if self.control_callback is not None:
                self.control_callback(position, index,self.qr_type)
            #update_waypoint(position, None, index)
            #qr_p.set_yaw_des_from_traj()
            return

        if callback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            # TODO(mereweth@jpl.nasa.gov) - attach control_callback here
            #update_waypoint(position, None, index, defer=True)
            #qr_p.set_yaw_des_from_traj()
            return

    def init_menu(self,waypoints):

        self.menu_handler = dict()

        self.num_waypoints = len(waypoints['x'][0,:])

        for i in range(0, self.num_waypoints ):
            key = str(i)
            self.menu_handler[key] = MenuHandler()
            # Insert waypoint button
            self.insert_entry_id = 1 # menu ID for insert
            h_insert = self.menu_handler[key].insert( "Insert Waypoint")
            self.menu_handler[key].insert( "Before", parent = h_insert, callback=self.insert_cb)
            self.menu_handler[key].insert( "After", parent = h_insert, callback=self.insert_cb)

            # Delete waypoint button
            self.menu_handler[key].insert( "Delete Waypoint", callback=self.delete_cb)


            # Menu to select waypoint type
            top_level = self.menu_handler[key].insert("Change Type")
            if self.qr_type is "main":
                # Checkbox menu items
                self.normal_option_ID = 6
                h_type = self.menu_handler[key].insert("Normal", parent = top_level, callback=self.change_type_cb)
                if self.check_tracking[i] == 0:
                    self.menu_handler[key].setCheckState(h_type, MenuHandler.CHECKED )
                else:
                    self.menu_handler[key].setCheckState(h_type, MenuHandler.UNCHECKED )

                self.entry_option_ID = 7
                h_type = self.menu_handler[key].insert("Entry", parent = top_level, callback=self.change_type_cb)
                if self.check_tracking[i] == 1 or self.check_tracking[i] == 3:
                    self.menu_handler[key].setCheckState(h_type, MenuHandler.CHECKED )
                else:
                    self.menu_handler[key].setCheckState(h_type, MenuHandler.UNCHECKED )

                self.exit_option_ID = 8
                h_type = self.menu_handler[key].insert("Exit", parent = top_level, callback=self.change_type_cb)
                if self.check_tracking[i] == 2 or self.check_tracking[i] == 3:
                    self.menu_handler[key].setCheckState(h_type, MenuHandler.CHECKED )
                else:
                    self.menu_handler[key].setCheckState(h_type, MenuHandler.UNCHECKED )
            elif self.qr_type is "entry":
                # Checkbox menu items
                h_type = self.menu_handler[key].insert("Take-off", parent = top_level, callback=self.change_type_cb)
                self.menu_handler[key].setCheckState(h_type, MenuHandler.CHECKED )
            elif self.qr_type is "exit":
                # Checkbox menu items
                h_type = self.menu_handler[key].insert("Landing", parent = top_level, callback=self.change_type_cb)
                self.menu_handler[key].setCheckState(h_type, MenuHandler.CHECKED )


        self.marker_server.applyChanges()

    def delete_cb( self, callback ):
        # get the index
        index = self.index_from_name(callback.marker_name)
        # rospy.loginfo("Deleting waypoint {}".format(index))

        if index <= self.entry_ID:
            # Reset entry if point deleted before or at the current entry waypoint
            self.add_entry_property(self.entry_ID-1)

        if index <= self.exit_ID:
            # Reset exit if point deleted before or at the current exit waypoint
            self.add_exit_property(self.exit_ID-1)

        # Command type for callback
        command_type = "delete"

        # Group information in data dict for callback
        data = dict()
        data['index'] = index
        data['entry_ID'] = self.entry_ID
        data['exit_ID'] = self.exit_ID

        # Callback
        self.menu_callback(command_type,data, self.qr_type)

    def insert_cb(self, callback):
        # get the index of the marker
        if callback.menu_entry_id == self.insert_entry_id + 1:
            # Insert before the current waypoint
            index = self.index_from_name(callback.marker_name)
        else:
            # Insert after the current waypoint
            index = self.index_from_name(callback.marker_name) + 1

        if index <= self.entry_ID:
            # Reset entry if point deleted before or at the current entry waypoint
            self.add_entry_property(self.entry_ID+1)

        if index <= self.exit_ID:
            # Reset exit if point deleted before or at the current exit waypoint
            self.add_exit_property(self.exit_ID+1)

        command_type = "insert"
        data = dict()
        data['index'] = index
        data['entry_ID'] = self.entry_ID
        data['exit_ID'] = self.exit_ID

        # Callback
        self.menu_callback(command_type,data,self.qr_type)

    def change_type_cb( self, callback ):

        # get the index
        index = self.index_from_name(callback.marker_name)
        key = str(index)

        # Check on
        if self.qr_type is not "main":
            return

        # handle for the menu entry
        handle = callback.menu_entry_id
        # DIfferent handle options to consider
        options = np.array([self.normal_option_ID, self.entry_option_ID, self.exit_option_ID])
        # Get checked state
        state = self.menu_handler[key].getCheckState( handle )

        if handle == self.entry_option_ID:
            # Entry
            if state == MenuHandler.CHECKED:
                self.remove_entry_property(index)
            else:
                self.add_entry_property(index)
            command_type = "change_entry"
        elif handle == self.exit_option_ID:
            # exit
            if state == MenuHandler.CHECKED:
                self.remove_exit_property(index)
            else:
                self.add_exit_property(index)
            command_type = "change_exit"
        else:
            # Normal input
            if state == MenuHandler.CHECKED:
                # do nothing
                return
            if self.check_tracking[index] == 3:
                # both
                self.remove_entry_property(index)
                self.remove_exit_property(index)
                command_type = "change_both"
            elif self.check_tracking[index] == 1:
                self.remove_entry_property(index)
                command_type = "change_entry"
            else:
                self.remove_exit_property(index)
                command_type = "change_exit"


        # Apply the changes
        self.menu_handler[key].reApply( self.marker_server )
        self.marker_server.applyChanges()

        # Send data through callback
        data = dict()
        data['index'] = index
        data['entry_ID'] = self.entry_ID
        data['exit_ID'] = self.exit_ID

        self.menu_callback(command_type,data,self.qr_type)

        # if self.menu_callback is not None:
        #     self.menu_callback(data)

    def add_entry_property(self,index):
        for i in range(0,self.num_waypoints):

            if i == index:
                # add entry property
                self.check_tracking[i] += 1

            elif self.check_tracking[i] == 1 or self.check_tracking[i] == 3:
                # Not Neutral - take away property
                self.check_tracking[i] -= 1

        self.entry_ID = index

    def add_exit_property(self,index):
        for i in range(0,self.num_waypoints):

            if i == index:
                # add entry property
                self.check_tracking[i] += 2

            elif self.check_tracking[i] >= 2:
                # Not Neutral - take away property
                self.check_tracking[i] -= 2

        self.exit_ID = index

    def remove_entry_property(self,index):
        self.check_tracking[index] -= 1

        # Reset entry ID to 0 TODO(bmorrell@jpl.nasa.gov) review if this is the best way to do it
        if index != 0:
            self.entry_ID = 0
            self.check_tracking[0] += 1
        else:
            self.entry_ID = 1
            self.check_tracking[1] += 1

    def remove_exit_property(self,index):
        self.check_tracking[index] -= 2

        # Reset entry ID to 0 TODO(bmorrell@jpl.nasa.gov) review if this is the best way to do it
        if index != 0:
            self.exit_ID = 0
            self.check_tracking[0] += 2
        else:
            self.exit_ID = 1
            self.check_tracking[1] += 2

    def stop(self):
        self.is_running = False

class AnimateWorker(QObject):
    finished = Signal() # class variable shared by all instances

    def __init__(self,
                 eval_callback,
                 dt = 0.01,
                 finished_callback=None,
                 node_name='animate_worker',
                 frame_id='animation',
                 parent_frame_id='local_origin',
                 marker_topic='animation_marker',
                 new_node=False,
                 slowdown=1,
                 parent=None):
        super(AnimateWorker, self).__init__(parent)

        if new_node:
            rospy.init_node(node_name, anonymous=True)

        if finished_callback is not None:
            self.finished.connect(finished_callback)

        #TODO(mereweth@jpl.nasa.gov) - how to shut down?
        self.is_running = True

        self.publish = True
        self.animate = False
        self.eval_callback = eval_callback
        self.t = 0.
        self.t_max = 0.
        self.dt = dt

        if slowdown <= 0.001: # prevent speeding up by more than 1000x
            slowdown = 1
        self.slowdown = slowdown

        self.R_old = None

        self.frame_id = frame_id
        self.parent_frame_id = parent_frame_id
        self.tf_br = tf.TransformBroadcaster()
        self.marker_pub = rospy.Publisher(marker_topic, MarkerArray, queue_size=1)
        self.refresh_marker()

        print("init animate")
        self.timer = QTimer()
        self.timer.setInterval(self.dt*1000*self.slowdown) # in milliseconds
        #self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.on_timer_callback)
        self.timer.start()

    def refresh_marker(self):
        quad_marker = build_quad_marker_template()
        quad_marker.id = 0
        quad_marker.header.frame_id = self.frame_id
        quad_marker.frame_locked = True
        quad_marker.action = quad_marker.ADD
        quad_marker.pose.orientation.w = 1
        quad_marker.pose.position.x = 0.0
        quad_marker.pose.position.y = 0.0
        quad_marker.pose.position.z = 0.0
        quad_marker.lifetime = rospy.Duration(0)

        marker_array = MarkerArray()
        marker_array.markers.append(quad_marker)
        self.marker_pub.publish(marker_array)

    def on_timer_callback(self):
        if not self.publish:
            return

        try:
            if self.animate:
                self.t += self.dt
                if self.t > self.t_max:
                    self.t = 0.
            else:

                self.t = 0.

            (t_max, x, y, z, qw, qx, qy, qz) = self.eval_callback(self.t)
            self.t_max = t_max


            self.tf_br.sendTransform((x,y,z),
                                     (qx, qy, qz, qw),
                                     rospy.Time.now(),
                                     self.frame_id,
                                     self.parent_frame_id)
        #except Exception as e:
            #print("Error in AnimateWorker")
            #print(e)
        except:
            # print("Unknown error in AnimateWorker")
            return

    def start_animate(self):
        print("Ros helper start animation")
        self.animate = True
        self.refresh_marker()

    def stop_animate(self):
        self.animate = False

    def start_publish(self):
        self.t = 0
        self.publish = True
        self.refresh_marker()

    def stop_publish(self):
        self.publish = False

class ObstacleControlWorker(QObject):
    finished = Signal() # class variable shared by all instances

    def __init__(self, control_callback,finished_callback=None,
                 node_name='obstacle_display_worker',
                 frame_id='local_origin',
                 marker_topic='obstacle_marker',
                 new_node=False,
                 parent=None):
        super(ObstacleControlWorker, self).__init__(parent)

        if new_node:
            rospy.init_node(node_name, anonymous=True)

        if finished_callback is not None:
            self.finished.connect(finished_callback)

        #TODO(mereweth@jpl.nasa.gov) - how to shut down?
        self.is_running = True

        self.control_callback = control_callback

        self.frame_id = frame_id

        self.marker_server = InteractiveMarkerServer(marker_topic)



    def make_obstacles(self, constraint_list):
        self.marker_server.clear()
        self.marker_server.applyChanges()

        for i in range(0, len(constraint_list) ):
            if constraint_list[i].constraint_type != "ellipsoid":
                # only plot ellipsoids
                continue

            if constraint_list[i].der != 0:
                # Only plot physical constraints
                continue



            # create an interactive marker
            int_marker = InteractiveMarker()
            int_marker.header.frame_id = self.frame_id
            int_marker.scale = 0.2
            int_marker.pose.position.x = constraint_list[i].x0[0]
            int_marker.pose.position.y = constraint_list[i].x0[1]
            int_marker.pose.position.z = constraint_list[i].x0[2]

            A = np.matrix(constraint_list[i].A)
            if hasattr(constraint_list[i],"rot_mat"):
                rot_mat = np.matrix(constraint_list[i].rot_mat)
                scale = rot_mat*A*rot_mat.T
            else:
                rot_mat = np.matrix(np.identity(3))
                scale = rot_mat*A*rot_mat.T
            q = transforms3d.quaternions.mat2quat(rot_mat)
            # NOTE: a differen convention with quaternions here
            int_marker.pose.orientation.w = q[0]
            int_marker.pose.orientation.x = q[1]
            int_marker.pose.orientation.y = q[2]
            int_marker.pose.orientation.z = q[3]

            int_marker.name = str(i)
            # create a non-interactive control which contains the box
            obstacle_control = InteractiveMarkerControl()
            obstacle_control.always_visible = True
            obstacle_control.orientation_mode = InteractiveMarkerControl.VIEW_FACING
            obstacle_control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
            obstacle_control.independent_marker_orientation = True

            obstacle_marker = self.build_obstacle_marker_template(constraint_list[i])

            obstacle_control.markers.append( obstacle_marker )
            int_marker.controls.append( obstacle_control )

            for ox,oy,oz,name in [(1,0,0,'move_x'),(0,1,0,'move_z'),(0,0,1,'move_y'),(1,0,0,'rotate_x'),(0,0,1,'rotate_y'),(0,1,0,'rotate_z')]:
                #not sure why z and y are swapped
                control = InteractiveMarkerControl()
                control.orientation.w = 1
                control.orientation.x = ox
                control.orientation.y = oy
                control.orientation.z = oz
                control.name = name
                if name[0]=='m':
                    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
                else:
                    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
                int_marker.controls.append(control)

            # callback probably needs to be in this thread for ROS
            # Then, that callback can use Qt signals and slots
            self.marker_server.insert( int_marker, self.obstacle_control_callback )


            self.marker_server.applyChanges()

    def update_obstacles(self, constraint_list, indices=None):

        n_obstacles = len(constraint_list)

        if indices is None:
            indices = range(0, n_obstacles)

        for i in indices:
            if constraint_list[i].constraint_type is "ellipsoid" and constraint_list[i].der == 0:
                pose = geometry_msgs.msg.Pose()
                pose.position.x = constraint_list[i].x0[0]
                pose.position.y = constraint_list[i].x0[1]
                pose.position.z = constraint_list[i].x0[2]

                A = np.matrix(constraint_list[i].A)
                if hasattr(constraint_list[i],"rot_mat"):
                    rot_mat = np.matrix(constraint_list[i].rot_mat)
                    scale = rot_mat*A*rot_mat.T
                else:
                    rot_mat = np.matrix(np.identity(3))
                    scale = rot_mat*A*rot_mat.T
                q = transforms3d.quaternions.mat2quat(rot_mat)
                # NOTE: a differen convention with quaternions here
                pose.orientation.w = q[0]
                pose.orientation.x = q[1]
                pose.orientation.y = q[2]
                pose.orientation.z = q[3]


                self.marker_server.setPose( str(i), pose )

                self.marker_server.applyChanges()

    def index_from_name(self, name):
        return int(re.search(r'\d+', name).group())

    def obstacle_control_callback(self, callback):
        #compute difference vector for this cube
        position = dict()
        position['x'] = callback.pose.position.x
        position['y'] = callback.pose.position.y
        position['z'] = callback.pose.position.z
        orientation = dict()
        orientation['w'] = callback.pose.orientation.w
        orientation['x'] = callback.pose.orientation.x
        orientation['y'] = callback.pose.orientation.y
        orientation['z'] = callback.pose.orientation.z
        index = self.index_from_name(callback.marker_name)

        #if index > len(qr_p.waypoints['x'][0,:]):
        #    return

        if callback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            print("Obstacle control callback for index " + str(index))

            if self.control_callback is not None:
                self.control_callback(position, orientation, index)
            #update_waypoint(position, None, index)
            #qr_p.set_yaw_des_from_traj()
            return

        if callback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            # TODO(mereweth@jpl.nasa.gov) - attach control_callback here
            #update_waypoint(position, None, index, defer=True)
            #qr_p.set_yaw_des_from_traj()
            return

    def build_obstacle_marker_template(self,constraint):
        obstacle_marker = Marker()
        obstacle_marker.type = obstacle_marker.SPHERE
        A = np.matrix(constraint.A)

        if hasattr(constraint,"rot_mat"):
            rot_mat = np.matrix(constraint.rot_mat)
            scale = rot_mat*A*rot_mat.T
        else:
            rot_mat = np.matrix(np.identity(3))
            scale = rot_mat*A*rot_mat.T

        # Size
        obstacle_marker.scale.x = np.sqrt(1/scale[0,0])
        obstacle_marker.scale.y = np.sqrt(1/scale[1,1])
        obstacle_marker.scale.z = np.sqrt(1/scale[2,2])
        obstacle_marker.color.r = 0.5
        obstacle_marker.color.g = 0.5
        obstacle_marker.color.b = 1
        obstacle_marker.color.a = 0.7
        return obstacle_marker

    def stop(self):
        self.is_running = False
