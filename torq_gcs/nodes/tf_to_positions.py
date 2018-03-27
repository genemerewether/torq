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

import numpy as np
import rospy
from tf.transformations import quaternion_from_euler

#from ssf_comm.msg import ExtState
import geometry_msgs.msg
from geometry_msgs.msg import QuaternionStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped


import tf

MSG_TIMEOUT = 1600000000
#MSG_TIMEOUT = 1.0

def fill_pose_msg(trans,rot):
    print("TF: to Position: ({})\nQuaternion (x,y,z,w): ({})\n\n".format(trans,rot))

    msg = geometry_msgs.msg.PoseStamped()

    msg.pose.position.x = trans[0]
    msg.pose.position.y = trans[1]
    msg.pose.position.z = trans[2]

    msg.pose.orientation.w = rot[3]
    msg.pose.orientation.x = rot[0]
    msg.pose.orientation.y = rot[1]
    msg.pose.orientation.z = rot[2]

    return msg


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.


    rospy.init_node('tf_to_pose_listener')

    pose_out_pub = rospy.Publisher("pose_stamped_vicon",PoseStamped,queue_size=1)

    tf_listener = tf.TransformListener()

    rate = rospy.Rate(2000.0) # - rate in hz
    while not rospy.is_shutdown():
        if tf_listener.frameExists("/world") and tf_listener.frameExists("/px4_pose_stamped"):
            print("in listener call")
            #     try:
            # (trans,rot) = tf_listener.lookupTransform("/local_origin","/vicon/torq250/torq250", rospy.Time(0))
            (trans,rot) = tf_listener.lookupTransform("/local_origin","/px4_pose_stamped", rospy.Time(0))
            # rospy.Time(0) gives the latest time
            # t = rospy.Time(0)
            msg = fill_pose_msg(trans,rot)

            pose_out_pub.publish(msg)
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     pass

        rate.sleep()


if __name__ == '__main__':
    listener()
