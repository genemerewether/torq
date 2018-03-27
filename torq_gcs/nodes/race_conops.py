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

import os
import signal
import sys

from px4_msgs.msg import PolyTraj as PolyTraj_msg

import rospkg, rospy
from python_qt_binding import loadUi
from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *
from python_qt_binding.QtWidgets import *

import torq_gcs

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.conops_widget = Splitter(self)
        self.setCentralWidget(self.conops_widget)

class Tabs(QTabWidget):
    def __init__(self, parent=None):
        super(Tabs, self).__init__(parent)

        self.setMinimumSize(400, 800)

        global_dict = dict()

        #self.spl = spline.SplineGUI(global_dict)
        #self.addTab(self.spl, "Spline fit")

        self.rdp_gui = torq_gcs.disc.line_simpl.RDPGUI(global_dict)
        self.addTab(self.rdp_gui, "RDP")#Ramer-Douglas-Peucker discretization")

        # self.snap = torq_gcs.plan.unco.QRPolyTrajGUI(global_dict)
        # self.addTab(self.snap, "UNCO")

        self.fs_planner = torq_gcs.plan.taco.FreespacePlannerGUI(global_dict)
        self.addTab(self.fs_planner, "TACO")#Freespace planner")

        self.esdf = torq_gcs.fsp.voxblox_widgets.ESDFGUI(global_dict)
        self.addTab(self.esdf, "ESDF")#Euclidean signed-distance field")

class Splitter(QSplitter):
    def __init__(self, parent):
        super(Splitter, self).__init__(parent)

        self.tabs = Tabs()

        config_file = os.path.join(rospkg.RosPack().get_path('torq_gcs'),
                                   'config', 'race_conops.rviz')
        self.grid = torq_gcs.viz.rviz_widgets.GridAndConfig(config_file)
        # QObject::moveToThread: Widgets cannot be moved to a new thread

        # TODO(mereweth@jpl.nasa.gov) - how to reuse combinations of widgets?
        self.addWidget(self.tabs)
        self.addWidget(self.grid.frame)

        # Important for performance; resize only on mouseup
        self.setOpaqueResize(False)

def main():
    app = QApplication( sys.argv )

    rospy.init_node('race_conops')

    rc = MainWindow()
    rc.show()

    return app.exec_()

if __name__=="__main__":
    sys.exit(main())
