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

from minsnap import utils

import rospkg
from python_qt_binding import loadUi
#from python_qt_binding.QtGui import *
#from python_qt_binding.QtCore import *
from python_qt_binding.QtWidgets import *

class SplineGUI(QWidget):
    def __init__(self, global_dict, parent=None):
        super(SplineGUI, self).__init__(parent)

        ui_file = os.path.join(rospkg.RosPack().get_path('torq_gcs'),
                               'resource', 'disc', 'Spline.ui')
        loadUi(ui_file, self)

        self.spline = None
        self.path_length = None
        self.in_waypoints = None

        # warn about overwriting?
        # if 'KEY' in global_dict.keys():
        global_dict['disc_out_waypoints'] = None
        #TODO(mereweth@jpl.nasa.gov) - custom signal for when update-function
        # is done? Is timing of signals guaranteed?
        global_dict['disc_updated_signal'] = self.smooth_button.clicked
        self.global_dict = global_dict

        self.load_waypoints_button.clicked.connect(self.on_load_waypoints_button_click)

        self.smooth_button.clicked.connect(self.on_smooth_button_click)

    def on_smooth_button_click(self):
        if self.in_waypoints is None:
            return

        (self.spline, self.path_length) = utils.spline_seed_traj(self.in_waypoints)
        # TODO(mereweth@jpl.nasa.gov) - discretize and set out_waypoints
        self.global_dict['disc_out_waypoints'] = self.in_waypoints

    def on_load_waypoints_button_click(self):
        filename = QFileDialog.getOpenFileName(self,
                                               'Import waypoints', '',
                                               "Waypoint YAML files (*.yaml)")
        if filename and len(filename)>0:
            filename = filename[0]
        else:
            print("Invalid file path")
            return

        try:
            self.in_waypoints = utils.load_waypoints(filename)
        except KeyError:
            print("Invalid file format")
            return
        except Exception as e:
            print("Unknown error loading waypoints from {}".format(filename))
            print(e)
            return

def main():
    app = QApplication( sys.argv )

    global_dict = dict()
    spline = SplineGUI(global_dict)

    # TODO(mereweth@jpl.nasa.gov) - pass gui object to config script
    try:
        import imp
        imp.load_source(torq_config,
                        os.path.abspath('~/Desktop/environments/344.py'))
        torq_config(spline)
    except:
        pass

    spline.show()

    # MarkerArray
    # Interactive Markers

    return app.exec_()

if __name__=="__main__":
    sys.exit(main())
