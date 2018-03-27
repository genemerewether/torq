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

import rospy, os
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
rospy.init_node('world_mesh')

mesh_file = './models/TORQ_250_041917_17k.ply'

publisher = rospy.Publisher('world_quad_marker', Marker, queue_size=100)

marker = Marker()
marker.id = 0
marker.type = marker.MESH_RESOURCE
marker.header.frame_id = "px4_pose_stamped"
marker.action = marker.ADD
#marker.action = marker.DELETE
marker.pose.orientation.x = 0
marker.pose.orientation.y = 0
marker.pose.orientation.z = 0
marker.pose.orientation.w = 1
marker.pose.position.x = 0
marker.pose.position.y = 0
marker.pose.position.z = 0
marker.scale.x = .001
marker.scale.y = .001
marker.scale.z = .001
marker.color.r = .2
marker.color.g = 1
marker.color.b = .2
marker.color.a = 1
marker.lifetime = rospy.Duration(0)
marker.frame_locked = True
marker.mesh_resource = 'file://' + os.path.abspath(mesh_file)
marker.mesh_use_embedded_materials = True

print("Loading %s" % marker.mesh_resource)
for j in range(5):
    publisher.publish( marker )
    rospy.sleep(1)
