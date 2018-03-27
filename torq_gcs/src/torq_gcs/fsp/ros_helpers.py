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
from sensor_msgs.msg import PointCloud2
import std_msgs
from visualization_msgs.msg import *
import sensor_msgs.point_cloud2 as pcl2
from interactive_markers.interactive_marker_server import *
from python_qt_binding.QtCore import QObject, Signal, QTimer

import transforms3d
from transforms3d import quaternions

import struct
import numpy
np = numpy

FSP_SPHERES_ID = 1

class WorldMeshWorker(QObject):
    finished = Signal() # class variable shared by all instances

    def __init__(self, finished_callback=None,
                 node_name='world_mesh_worker',
                 frame_id='local_origin',
                 marker_topic='world_mesh_marker',
                 refresh_ms=None,
                 new_node=False,
                 parent=None):
        super(WorldMeshWorker, self).__init__(parent)

        if new_node:
            rospy.init_node(node_name, anonymous=True)

        if finished_callback is not None:
            self.finished.connect(finished_callback)

        #TODO(mereweth@jpl.nasa.gov) - how to shut down?
        self.is_running = True

        self.frame_id = frame_id
        self.marker_pub = rospy.Publisher(marker_topic, Marker,
                                          queue_size=100)

        self.file_resource = None
        # if marker disappears, this will refresh it
        # Rviz is only slow to load a mesh the first time
        if refresh_ms is not None:
            self.timer = QTimer()
            self.timer.setInterval(refresh_ms) # in milliseconds
            #self.timer.setTimerType(Qt.PreciseTimer)
            self.timer.timeout.connect(self.on_timer_callback)
            self.timer.start()

    def publish_marker(self, file_resource):
        self.file_resource = file_resource

        marker = Marker()
        marker.id = 0
        marker.ns = 'world_mesh'
        marker.type = marker.MESH_RESOURCE
        marker.header.frame_id = self.frame_id
        marker.action = marker.ADD
        #marker.action = marker.DELETE
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        #marker.color.r = 1
        #marker.color.g = 1
        #marker.color.b = 1
        #marker.color.a = 1
        marker.lifetime = rospy.Duration(0)
        marker.mesh_resource = file_resource
        marker.mesh_use_embedded_materials = True

        self.marker_pub.publish(marker)

    def stop(self):
        self.is_running = False

    def on_timer_callback(self):
        if self.file_resource is None:
            return
        self.publish_marker(self.file_resource)

class IntensityPCLWorker(QObject):
    finished = Signal() # class variable shared by all instances

    def __init__(self, finished_callback=None,
                 node_name='intensity_pcl_worker',
                 frame_id='local_origin',
                 marker_topic='fsp_distance_pcl',
                 new_node=False,
                 parent=None):
        super(IntensityPCLWorker, self).__init__(parent)

        if new_node:
            rospy.init_node(node_name, anonymous=True)

        if finished_callback is not None:
            self.finished.connect(finished_callback)

        #TODO(mereweth@jpl.nasa.gov) - how to shut down?
        self.is_running = True

        self.frame_id = frame_id
        self.pcl_pub = rospy.Publisher(marker_topic, PointCloud2,
                                          queue_size=100)

        self.define_fields()

    def define_fields(self):
        self.fields = list()

        field = pcl2.PointField()
        field.name = "x"
        field.offset = 0
        field.datatype = pcl2.PointField.FLOAT32
        field.count = 1

        self.fields.append(field)

        field = pcl2.PointField()
        field.name = "y"
        field.offset = 4
        field.datatype = pcl2.PointField.FLOAT32
        field.count = 1

        self.fields.append(field)

        field = pcl2.PointField()
        field.name = "z"
        field.offset = 8
        field.datatype = pcl2.PointField.FLOAT32
        field.count = 1

        self.fields.append(field)

        field = pcl2.PointField()
        field.name = "intensity"
        field.offset = 12
        field.datatype = pcl2.PointField.FLOAT32
        field.count = 1

        self.fields.append(field)

    def publish_point_cloud(self, x, y, z, i):
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        z = np.array(z).flatten()
        i = np.array(i).flatten()

        x_shp = np.shape(x)
        y_shp = np.shape(y)
        z_shp = np.shape(z)
        i_shp = np.shape(i)

        if x_shp != y_shp or x_shp != z_shp or x_shp != i_shp:
            return

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id

        cloud_points = zip(x, y, z, i)

        #create pcl from points
        try:
            cloud = pcl2.create_cloud(header,
                                      self.fields,
                                      cloud_points)
        except struct.error as e:
            print(e)
            return
        except Exception as e:
            print(e)
            return
        except:
            print("Unknown exception while creating point cloud")
            return

        cloud.is_dense = True;
        self.pcl_pub.publish(cloud)

    def stop(self):
        self.is_running = False

class PlaneControlWorker(QObject):
    finished = Signal() # class variable shared by all instances

    def __init__(self,
                 control_callback=None,
                 finished_callback=None,
                 node_name='plane_control_worker',
                 frame_id='local_origin',
                 control_marker_topic='plane_control',
                 plane_marker_topic='plane_marker',
                 new_node=False,
                 parent=None):

        super(PlaneControlWorker, self).__init__(parent)

        if new_node:
            rospy.init_node(node_name, anonymous=True)

        if finished_callback is not None:
            self.finished.connect(finished_callback)

        # create an interactive marker server on the topic namespace simple_marker
        self.server = InteractiveMarkerServer(control_marker_topic)

        # TOD maybe set these to none and have checks?
        self.scale_global = None
        self.pos_global = None
        self.rot_global = None

        self.axis_keys = ('x','y','z')
        self.scale_control_scale = 0.4
        self.plane_control_scale = 2.0
        self.expand_factor = 1.0

        self.min_scale = 0.1

        self.is_running = True

        self.control_callback = control_callback

        self.frame_id = frame_id


        # self.make6DoFMarker()
        #
        # self.makeScaleMarkers()
        #
        # # 'commit' changes and send to all clients
        # self.server.applyChanges()

    def processFeedback(self, feedback):
        p = feedback.pose.position
        o = feedback.pose.orientation
        # print feedback.marker_name + " is now at " + str(p.x) + ", " + str(p.y) + ", " + str(p.z)
        # print "Orientation is "+ str(o.w) + ", " + str(o.x) + ", " + str(o.y) + ", " + str(o.z)


        if feedback.marker_name[0] == 's':
            if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
                # Scale control
                axis = feedback.marker_name[14]
                if feedback.marker_name[-3:] == "pos":
                    sign = 1
                else:
                    sign = -1

                # transform position back to body frame
                position = np.array([p.x,p.y,p.z])
                position, q = self.transformGlobal2Body(axis, position, self.rot_global, self.pos_global)

                # Work out the new scale
                index_dict = dict(x=0,y=1,z=2)
                self.scale_global[index_dict[axis]] = position[index_dict[axis]]*sign*2.0/self.expand_factor

                if self.scale_global[index_dict[axis]] < self.min_scale:
                    # Cap at a minimum
                    self.scale_global[index_dict[axis]] = self.min_scale

                # Change the plane marker
                self.make6DoFMarker()

                # # Callback
                # self.control_callback(self.pos_global,self.rot_global, self.scale_global)

        if feedback.marker_name[0] == 'p':
            # server.erase("scale_control_x")
            if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
                # Plane control
                # update position
                self.pos_global = np.array([p.x,p.y,p.z])
                # Update orientation
                self.rot_global = np.array([o.w,o.x,o.y,o.z])

                # Update scale markers
                self.makeScaleMarkers()

                # # Callback
                # self.control_callback(self.pos_global,self.rot_global, self.scale_global)


        self.server.applyChanges()

    def transformBody2Global(self, axis, position, orientation, centre):

        # Rotate
        if type(orientation) is np.ndarray:
            q = orientation
        else:
            q = np.array([orientation.w,orientation.x,orientation.y,orientation.z])

        R = transforms3d.quaternions.quat2mat(q)
        position = R.dot(position)

        # Translate
        if type(centre) is np.ndarray:
            position +=  centre
        else:
            position +=  np.array([centre.x,centre.y,centre.z])

        return position, q

    def transformGlobal2Body(self, axis, position, orientation, centre):

        # Translate (un-translate)
        print("Centre is {}".format(centre))
        if type(centre) is np.ndarray:
            position -=  centre
        else:
            position -=  np.array([centre.x,centre.y,centre.z])

        # Rotate (inverse)
        if type(orientation) is np.ndarray:
            q = orientation
        else:
            q = np.array([orientation.w,orientation.x,orientation.y,orientation.z])

        R = transforms3d.quaternions.quat2mat(q)
        position = R.T.dot(position)

        return position, q

    def makeBox(self, msg, scale):
        marker = Marker()

        marker.type = Marker.CUBE
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 0.25

        return marker

    def makeBoxControl(self, msg, scale):
        control =  InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append( self.makeBox(msg, scale) )
        msg.controls.append( control )
        return control

    def make6DoFMarker(self):#, position, orientation=np.array([1,0,0,0.])):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.frame_id
        int_marker.pose.position.x = self.pos_global[0]
        int_marker.pose.position.y = self.pos_global[1]
        int_marker.pose.position.z = self.pos_global[2]
        int_marker.pose.orientation.w = self.rot_global[0]
        int_marker.pose.orientation.x = self.rot_global[1]
        int_marker.pose.orientation.y = self.rot_global[2]
        int_marker.pose.orientation.z = self.rot_global[3]
        int_marker.scale = self.plane_control_scale

        int_marker.name = "plane control"

        scale = self.scale_global
        self.makeBoxControl(int_marker, scale)
        int_marker.controls[0].interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D

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

        self.server.insert(int_marker, self.processFeedback)

    def makeScaleMarkers(self):

        vector = dict(x=np.array([self.scale_global[0]/2*self.expand_factor,0,0]),y=np.array([0,self.scale_global[1]/2*self.expand_factor,0]),z=np.array([0,0,self.scale_global[2]/2*self.expand_factor]))
        oxyz = dict(x=(1,0,0),y=(0,0,1),z=(0,1,0))

        for axis in ('x','y','z'):
            for sign in (-1,1):

                # Body frame position
                position = vector[axis]*sign

                # Transform
                position, q  = self.transformBody2Global(axis, position, self.rot_global, self.pos_global)

                # Create marker
                int_marker = InteractiveMarker()
                int_marker.header.frame_id = self.frame_id
                int_marker.pose.position.x = position[0]
                int_marker.pose.position.y = position[1]
                int_marker.pose.position.z = position[2]
                int_marker.pose.orientation.w = q[0]
                int_marker.pose.orientation.x = q[1]
                int_marker.pose.orientation.y = q[2]
                int_marker.pose.orientation.z = q[3]
                int_marker.scale = self.scale_control_scale

                if sign == 1:
                    int_marker.name = "scale_control_" + axis + "_pos"
                else:
                    int_marker.name = "scale_control_" + axis + "_neg"

                control = InteractiveMarkerControl()
                control.orientation.w = 1
                control.orientation.x = oxyz[axis][0]
                control.orientation.y = oxyz[axis][1]
                control.orientation.z = oxyz[axis][2]
                control.name = "move_" + axis + "+" + str(sign)
                control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
                control.orientation_mode = InteractiveMarkerControl.INHERIT#InteractiveMarkerControl.FIXED # SO the axis changes with the orientation
                int_marker.controls.append(control)

                self.server.insert(int_marker, self.processFeedback)

class FreeSpheresWorker(QObject):
    finished = Signal() # class variable shared by all instances

    def __init__(self, finished_callback=None,
                 node_name='free_spheres_worker',
                 frame_id='local_origin',
                 marker_topic='free_spheres_marker',
                 new_node=False,
                 parent=None):
        super(FreeSpheresWorker, self).__init__(parent)

        if new_node:
            rospy.init_node(node_name, anonymous=True)

        if finished_callback is not None:
            self.finished.connect(finished_callback)

        #TODO(mereweth@jpl.nasa.gov) - how to shut down?
        self.is_running = True

        self.n_spheres = 0

        self.frame_id = frame_id
        self.marker_pub = rospy.Publisher(marker_topic, MarkerArray,
                                          queue_size=100)

    def publish_spheres(self, x, y, z, r):
        spheres_marker_array = []
        if len(x) < self.n_spheres:
            num = self.n_spheres
        else:
            num = len(x)
        for i in range(0,num):
            # clear the old markers
            spheres_marker = Marker()
            spheres_marker.header.frame_id = self.frame_id
            spheres_marker.id = FSP_SPHERES_ID + i
            spheres_marker.action = spheres_marker.DELETE
            spheres_marker_array.append(spheres_marker)

            if i < len(x):
                spheres_marker = Marker()
                spheres_marker.header.frame_id = self.frame_id
                spheres_marker.type = spheres_marker.SPHERE
                spheres_marker.id = FSP_SPHERES_ID + i
                spheres_marker.action = spheres_marker.ADD
                spheres_marker.scale.x = 2*r[i]
                spheres_marker.scale.y = 2*r[i]
                spheres_marker.scale.z = 2*r[i]
                spheres_marker.color.a = 0.5
                spheres_marker.color.r = 1
                spheres_marker.color.g = 1
                spheres_marker.color.b = 1
                spheres_marker.pose.orientation.w = 1
                spheres_marker.pose.position.x = x[i]
                spheres_marker.pose.position.y = y[i]
                spheres_marker.pose.position.z = z[i]
                spheres_marker_array.append(spheres_marker)

        self.n_spheres = len(x)
        self.marker_pub.publish(spheres_marker_array)
