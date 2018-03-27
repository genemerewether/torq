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

from torq_gcs.viz.ros_helpers import CameraPlacementWorkerNoQt

def main():
    cpw = CameraPlacementWorkerNoQt(new_node=True)
    cpw.spin()

if __name__ == '__main__':
    main()
