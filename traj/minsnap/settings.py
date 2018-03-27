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

# GLOBAL CONSTANTS
STATE_INFO_TOKEN = "state_info"
FIELD_ORDER_TOKEN = "field_order"

TANGO_MAP_T_BODY_FIELDS = ['timestamp [ns]',
                           'pos.x [m]',
                           'pos.y [m]',
                           'pos.z [m]',
                           'rot.x',
                           'rot.y',
                           'rot.z',
                           'rot.w']

TIME_NZ_EPS = 0.01
