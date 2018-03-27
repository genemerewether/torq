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
from geometry_msgs.msg import Twist, Point, Vector3
from view_controller_msgs.msg import CameraPlacement
import numpy as np
from python_qt_binding.QtCore import QObject, Signal, QTimer

import transforms3d

class CameraPlacementWorkerNoQt():
    def __init__(self,
                 tfo_in_camera_yaw_frame=True,
                 node_name='camera_placement_worker',
                 camera_placement_topic='rviz/camera_placement',
                 twist_input_topic='spacenav/twist',
                 twist_input_rate=1,
                 camera_speed = dict(trans = 0.25,
                                     zoom  = 0.1,
                                     pitch = 0.005,
                                     yaw   = 0.01),
                 start_eye = np.array([5,5,5], dtype='float64'),
                 start_focus = np.array([0,0,0], dtype='float64'),
                 new_node=False):

        if new_node:
            rospy.init_node(node_name, anonymous=True)

        self.eye = np.array(start_eye)
        self.focus = np.array(start_focus)
        self.up = np.array([0, 0, 1], dtype='float64')

        # TODO(mereweth@jpl.nasa.gov) rviz_animated_view_controller doesn't
        # respond well to simultaneous rotation and translation. Is this a
        # limitation of Ogre/Rviz?
        self.either_rot_or_trans = True

        self.tfo_in_camera_yaw_frame = tfo_in_camera_yaw_frame
        self.twist_input_rate = twist_input_rate
        self.camera_speed = camera_speed

        self.camera_placement_pub = rospy.Publisher(camera_placement_topic,
                                                    CameraPlacement,
                                                    queue_size=100)

        self.twist_sub = rospy.Subscriber(twist_input_topic, Twist,
                                          self.callback_twist)

    def spin(self):
        rospy.spin()

    def callback_twist(self, data):
        cam = CameraPlacement()
        cam.interpolation_mode = cam.SPHERICAL

        offset = self.eye - self.focus

        pitch_rot = transforms3d.quaternions.axangle2quat(np.array([0,
                                                                    data.angular.y,
                                                                    0]),
                                                          self.camera_speed['pitch'])
        yaw_rot = transforms3d.quaternions.axangle2quat(np.array([0, 0,
                                                                  data.angular.z]),
                                                        self.camera_speed['yaw'])

        trans = self.camera_speed['trans'] * np.array([data.linear.x,
                                                       data.linear.y,
                                                       data.linear.z],
                                                      dtype='float64')

        # Transform transformations from world to camera yaw-aligned frame
        if self.tfo_in_camera_yaw_frame:
            view_yaw = np.arctan2(offset[1], offset[0])
            view_yaw_rot = [np.cos(view_yaw/2), 0, 0, np.sin(view_yaw/2)]

            new_trans = transforms3d.quaternions.rotate_vector(trans, view_yaw_rot)
            if not np.any(np.isnan(new_trans)):
                # make all three axes consistent
                new_trans[0] = -new_trans[0]
                new_trans[1] = -new_trans[1]
                trans = new_trans

        # don't allow rotation if translating
        if not self.either_rot_or_trans or np.allclose(trans, np.zeros(3)):
            # make all three axes consistent

            #new_offset = transforms3d.quaternions.rotate_vector(offset, pitch_rot)
            new_offset = transforms3d.quaternions.rotate_vector(offset, yaw_rot)
            if not np.any(np.isnan(new_offset)):
                offset = new_offset

            cam.allow_free_yaw_axis = False
            #NOTE(mereweth@jpl.nasa.gov) - set allow_free_yaw_axis True and
            # change up vector to spin about camera axis

        self.focus += trans
        self.eye = self.focus + offset

        cam.focus.point.x = self.focus[0]
        cam.focus.point.y = self.focus[1]
        cam.focus.point.z = self.focus[2]

        cam.eye.point.x = self.eye[0]
        cam.eye.point.y = self.eye[1]
        cam.eye.point.z = self.eye[2]

        cam.up.vector.x = self.up[0]
        cam.up.vector.y = self.up[1]
        cam.up.vector.z = self.up[2]

        cam.time_from_start = rospy.Duration(0) #1.0 / self.twist_input_rate)

        self.camera_placement_pub.publish(cam)

class CameraPlacementWorker(QObject):
    finished = Signal() # class variable shared by all instances

    def __init__(self, finished_callback=None,
                 parent=None,
                 **kwargs):
        super(CameraPlacementWorker, self).__init__(parent)

        if finished_callback is not None:
            self.finished.connect(finished_callback)

        self.cpw = CameraPlacementWorkerNoQt(**kwargs)

    def stop(self):
        self.is_running = False
