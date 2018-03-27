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

__author__ = "Gene Merewether"
__email__ = "mereweth@jpl.nasa.gov"


# Import statements
# =================

import yaml
import os
import sys
import argparse
import time
import pickle

import numpy.linalg
np = numpy

import rospy

from geometry_msgs.msg import PoseStamped

# TODO(mereweth@jpl.nasa.gov) - is this Pythonic?
try:
    from minsnap.settings import STATE_INFO_TOKEN, FIELD_ORDER_TOKEN
    HAS_MINSNAP = True
except ImportError:
    HAS_MINSNAP = False
    # GLOBAL CONSTANTS
    STATE_INFO_TOKEN = "state_info"
    FIELD_ORDER_TOKEN = "field_order"


class Recorder(object):
    def __init__(self, filename, disc=0.1, stale_time=0.1, linear=True):
        self.disc = disc
        # unused for now; determines discretization method
        self.linear = linear
        self.pose_last = None
        self.filename = filename
        self.stale_time = stale_time
        filbase, filext = os.path.splitext(self.filename)

        i = 0
        while os.path.exists(self.filename):
            print("{} already exists; appending suffix _{}".format(
                self.filename, i))
            self.filename = "{}_{}{}".format(filbase, i, filext)
            i += 1

        self.filp = None

    def __enter__(self):
        self.filp = open(self.filename, 'a')
        return self

    def __exit__(self, *args):
        self.filp.close()

    # conscious decision to log data as we go
    def posestamped_callback(self, msg):

        t_now = time.time()
        t_msg = msg.header.stamp.to_sec()

        if t_now - t_msg > self.stale_time:
            return

        r = np.array([msg.pose.position.x,
                      msg.pose.position.y,
                      msg.pose.position.z])

        if self.pose_last == None:
            r_last = np.array([0, 0, 0])
        else:
            r_last = np.array([self.pose_last.position.x,
                               self.pose_last.position.y,
                               self.pose_last.position.z])

        q_w = msg.pose.orientation.w
        q_x = msg.pose.orientation.x
        q_y = msg.pose.orientation.y
        q_z = msg.pose.orientation.z

        if (self.pose_last == None) or (self.linear and
                                        (np.linalg.norm(r - r_last) > self.disc)):

            # TODO(mereweth@jpl.nasa.gov) - check how fragile this
            # "write as you go" is

            if (self.pose_last == None):
                with self:
                    data = dict()
                    data[STATE_INFO_TOKEN] = dict()
                    data[STATE_INFO_TOKEN]['t'] = dict(
                        units="ROS2 time, seconds")
                    data[STATE_INFO_TOKEN]['x'] = dict(units=["meters",
                                                              "meters", "meters", "quaternion w", "quaternion x",
                                                              "quaternion y", "quaternion z"])
                    data[STATE_INFO_TOKEN]['x'][FIELD_ORDER_TOKEN] = ["x", "y", "z",
                                                                      "q_w", "q_x", "q_y", "q_z"]
                    self.filp.write(yaml.dump(data, default_flow_style=None))
                    self.filp.write("states:\n")

            data = [dict(t=t_msg,
                         x=r.tolist() + [q_w, q_x, q_y, q_z])]

            print("Writing {} to {}".format(data, self.filename))

            with self:
                self.filp.write(yaml.dump(data, default_flow_style=None))
            self.pose_last = msg.pose


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-f', '--file', default="~/snappy/traj/share/sample_data/waypoints.yaml",
        help='filename to log waypoints to')

    args = parser.parse_args(argv)
    args.file = os.path.expanduser(args.file)

    node = rospy.init_node('spatial_disc', anonymous=True)

    rec = Recorder(args.file)
    posestamped_sub = rospy.Subscriber('pose_stamped_out', PoseStamped,
                                       rec.posestamped_callback)

    assert posestamped_sub  # prevent unused warning

    rospy.spin()

if __name__ == '__main__':
    main()
