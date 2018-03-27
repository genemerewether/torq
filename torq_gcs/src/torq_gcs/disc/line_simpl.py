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

import rdp
import numpy
np = numpy

import rospkg
from python_qt_binding import loadUi
from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *
from python_qt_binding.QtWidgets import *

class RDPGUI(QWidget):
    def __init__(self, global_dict, parent=None):
        super(RDPGUI, self).__init__(parent)

        ui_file = os.path.join(rospkg.RosPack().get_path('torq_gcs'),
                               'resource', 'disc', 'RDP.ui')
        loadUi(ui_file, self)

        self.in_waypoints = None

        # warn about overwriting?
        # if 'KEY' in global_dict.keys():
        global_dict['disc_out_waypoints'] = None

        #TODO(mereweth@jpl.nasa.gov) - custom signal for when update-function
        # is done? Is timing of signals guaranteed?
        global_dict['disc_updated_signal'] = self.simplify_button.clicked

        global_dict['full_waypoints'] = None
        global_dict['disc_mask'] = None

        self.global_dict = global_dict

        self.load_waypoints_button.clicked.connect(self.on_load_waypoints_button_click)
        self.save_waypoints_button.clicked.connect(self.on_save_waypoints_button_click)
        self.simplify_button.clicked.connect(self.on_simplify_button_click)

        double_validator = QDoubleValidator(parent=self.epsilon_line_edit)
        double_validator.setBottom(0)
        self.epsilon_line_edit.setValidator(double_validator)

    def epsilon(self):
        try:
            return float(self.epsilon_line_edit.text())
        except ValueError as e:
            print(e)
            self.epsilon_line_edit.text()
            self.epsilon_line_edit.setText("0")
            return 0

    def on_simplify_button_click(self, checked=False, epsilon=None):
        if self.in_waypoints is None:
            return

        if epsilon is None:
            ep = self.epsilon()
        else:
            ep = epsilon

        temp_waypoints = np.c_[self.in_waypoints['x'],
                                       self.in_waypoints['y'],
                                       self.in_waypoints['z']]

        waypoint_mask = rdp.rdp(temp_waypoints,
                                 epsilon=ep,
                                 return_mask=True)

        out_waypoints = temp_waypoints[waypoint_mask]

        self.global_dict['disc_out_waypoints'] = dict(yaw=None)
        self.global_dict['disc_out_waypoints']['x'] = out_waypoints[:, 0]
        self.global_dict['disc_out_waypoints']['y'] = out_waypoints[:, 1]
        self.global_dict['disc_out_waypoints']['z'] = out_waypoints[:, 2]

        self.global_dict['disc_mask'] = np.where(waypoint_mask)[0]


        if ep == 0.:
            self.global_dict['disc_out_waypoints']['yaw'] = self.in_waypoints['yaw']

        print("Simplified waypoints")

    def on_load_waypoints_button_click(self, checked=False, filename=None):
        if filename is None:
            filename = QFileDialog.getOpenFileName(self,
                                                   'Import waypoints', #path,
                                                   "Waypoint YAML files (*.yaml)")
            if filename and len(filename)>0:
                filename = filename[0]
            else:
                print("Invalid file path")
                return

        try:
            self.in_waypoints = utils.load_waypoints(filename)
            self.global_dict['full_waypoints'] = self.in_waypoints
            print("Loaded waypoints from {}".format(filename))
        except KeyError:
            print("Invalid file format")
            return
        except Exception as e:
            print("Unknown error loading waypoints from {}".format(filename))
            print(e)
            return

    def on_save_waypoints_button_click(self, checked=False, filename=None):
        if self.global_dict['disc_out_waypoints'] is None:
            print("No waypoints to save")
            return

        if filename is None:
            filename = QFileDialog.getSaveFileName(self,
                                                   'Export waypoints', #path,
                                                   "Waypoint YAML files (*.yaml)")
            if filename and len(filename)>0:
                filename = filename[0]
            else:
                print("Invalid file path")
                return

        try:
            utils.save_waypoints(self.global_dict['disc_out_waypoints'], filename)
            print("Saved waypoints to {}".format(filename))
        except KeyError:
            print("Invalid file format")
            return
        except Exception as e:
            print("Unknown error saving waypoints to {}".format(filename))
            print(e)
            return

def main():
    from torq_gcs.disc.line_simpl import RDPGUI

    app = QApplication( sys.argv )

    global_dict = dict()
    rdp_gui = RDPGUI(global_dict)

    # TODO(mereweth@jpl.nasa.gov) - pass gui object to config script
    try:
        import imp
        conf = imp.load_source('torq_config',
                        os.path.expanduser('~/Desktop/environments/344.py'))
        conf.torq_config(rdp_gui)
    except Exception as e:
        print("Error in config script")
        print(e)

    rdp_gui.show()

    # MarkerArray
    # Interactive Markers

    return app.exec_()

if __name__=="__main__":
    sys.exit(main())
