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
from geometry_msgs.msg import QuaternionStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped

import tf

#MSG_TIMEOUT = 1600000000
MSG_TIMEOUT = 1.0

class Node:
  def __init__(self, id, qw, qx, qy, qz, x, y, z):
    self.x = x
    self.y = y
    self.z = z
    self.qw = qw
    self.qx = qx
    self.qy = qy
    self.qz = qz
    self.id = id
  def set_cov(self, cov):
    self.cov = np.array(cov)

  def __str__(self):
    return "ID: {0}\nPosition: ({1}, {2}, {3})\nQuaternion (x,y,z,w): ({4}, {5}, {6}, {7})\n\n".format(self.id,self.x,self.y,self.z,self.qx,self.qy,self.qz,self.qw)


def tandem_to_node(id, tandem_yaw, tandem_position):
    quaternion = quaternion_from_euler(0,0,tandem_yaw)
    node = Node(id,quaternion[3] ,quaternion[0] ,quaternion[1] ,quaternion[2] ,tandem_position[0] ,tandem_position[1] ,tandem_position[2] )

    return node


def publish_tf(node, origin):
  print node
  br = tf.TransformBroadcaster()
  br.sendTransform((node.x, node.y, node.z),
                   (node.qx, node.qy, node.qz, node.qw),
                   rospy.Time.now(),
                   node.id,
                   origin)


def callback_pose_stamped(data):
    if (rospy.get_time() - data.header.stamp.to_sec()) > MSG_TIMEOUT:
      print("discarding late pose_stamped message")
      return

    t_x = data.pose.position.x
    t_y = data.pose.position.y
    t_z = data.pose.position.z

    q_w = data.pose.orientation.w
    q_x = data.pose.orientation.x
    q_y = data.pose.orientation.y
    q_z = data.pose.orientation.z


    px4_pose = Node("px4_pose_stamped",q_w,q_x,q_y,q_z, t_x, t_y, t_z)
    publish_tf(px4_pose, "local_origin")

def callback_test_vicon(data):
    if (rospy.get_time() - data.header.stamp.to_sec()) > MSG_TIMEOUT:
      print("discarding late pose_stamped message")
      return

    t_x = data.pose.position.x + 1.0
    t_y = data.pose.position.y
    t_z = data.pose.position.z

    q_w = data.pose.orientation.w
    q_x = data.pose.orientation.x
    q_y = data.pose.orientation.y
    q_z = data.pose.orientation.z


    px4_pose = Node("px4_test_vicon",q_w,q_x,q_y,q_z, t_x, t_y, t_z)
    publish_tf(px4_pose, "local_origin")


def callback_pose_stamped_setpoint(data):
    if (rospy.get_time() - data.header.stamp.to_sec()) > MSG_TIMEOUT:
      print("discarding late pose_stamped setpoint message")
      return

    t_x = data.pose.position.x
    t_y = data.pose.position.y
    t_z = data.pose.position.z

    q_w = data.pose.orientation.w
    q_x = data.pose.orientation.x
    q_y = data.pose.orientation.y
    q_z = data.pose.orientation.z


    px4_setpoint = Node("px4_pose_stamped_setpoint",q_w,q_x,q_y,q_z, t_x, t_y, t_z)
    publish_tf(px4_setpoint, "local_origin")


def callback_quaternion_stamped_setpoint(data):
    if (rospy.get_time() - data.header.stamp.to_sec()) > MSG_TIMEOUT:
      print("discarding late quaternion_stamped setpoint message")
      return

    # NOTE(mereweth@jpl.nasa.gov) - could change this to current/commanded drone position
    t_x = 0
    t_y = 0
    t_z = 0

    q_w = data.quaternion.w
    q_x = data.quaternion.x
    q_y = data.quaternion.y
    q_z = data.quaternion.z


    px4_setpoint = Node("px4_quaternion_stamped_setpoint",q_w,q_x,q_y,q_z, t_x, t_y, t_z)
    publish_tf(px4_setpoint, "local_origin")


def callback_ssf_pose(data):
    if (rospy.get_time() - data.header.stamp.to_sec()) > 0.1:
      return

    t_x = data.pose.pose.position.x
    t_y = data.pose.pose.position.y
    t_z = data.pose.pose.position.z

    q_w = data.pose.pose.orientation.w
    q_x = data.pose.pose.orientation.x
    q_y = data.pose.pose.orientation.y
    q_z = data.pose.pose.orientation.z


    ssf_pose = Node("SSF_Pose",q_w,q_x,q_y,q_z, t_x, t_y, t_z)
    publish_tf(ssf_pose, "local_origin")


def callback_ssf_pose_vel(data):
    if (rospy.get_time() - data.header.stamp.to_sec()) > 0.1:
      return

    t_x = data.pose.position.x
    t_y = data.pose.position.y
    t_z = data.pose.position.z

    q_w = data.pose.orientation.w
    q_x = data.pose.orientation.x
    q_y = data.pose.orientation.y
    q_z = data.pose.orientation.z

    ssf_pose = Node("SSF_ExtState",q_w,q_x,q_y,q_z, t_x, t_y, t_z)
    publish_tf(ssf_pose, "local_origin")


def callback_vicon(data):

    t_x = data.transform.translation.x
    t_y = data.transform.translation.y
    t_z = data.transform.translation.z

    q_w = data.transform.rotation.w
    q_x = data.transform.rotation.x
    q_y = data.transform.rotation.y
    q_z = data.transform.rotation.z

    vicon_position = Node("vicon/torq250/torq250",q_w,q_x,q_y,q_z, t_x, t_y, t_z)
    publish_tf(vicon_position, "local_origin")

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('tf_visualization')

    #rospy.Subscriber("ssf_ext_state_input", ExtState, callback_ssf_pose_vel)

    rospy.Subscriber("pose_stamped_out", PoseStamped, callback_pose_stamped)

    # rospy.Subscriber("pose_stamped_vicon", PoseStamped, callback_test_vicon)

    rospy.Subscriber("pose_stamped_setpoint_out", PoseStamped, callback_pose_stamped_setpoint)

    rospy.Subscriber("quaternion_stamped_setpoint_out", QuaternionStamped, callback_quaternion_stamped_setpoint)

    # rospy.Subscriber("/vicon/torq250/torq250", TransformStamped, callback_vicon)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()



if __name__ == '__main__':
    listener()
