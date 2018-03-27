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

import sys
import os

from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *
from python_qt_binding.QtWidgets import *

## Finally import the RViz bindings themselves.
import rviz

import rospkg

class GridAndConfig(QWidget):
    def __init__(self, config_file, parent=None):
        super(GridAndConfig, self).__init__(parent)
        self.frame = rviz.VisualizationFrame()
        self.frame.setSplashPath( "" )
        self.frame.initialize()

        reader = rviz.YamlConfigReader()
        config = rviz.Config()
        reader.readFile( config, config_file )
        self.frame.load( config )

# TODO(mereweth@jpl.nasa.gov) - how do we ignore parts of a config file
# to remove only the dockable widgets?
class GridOnly(GridAndConfig):
    def __init__(self, config_file):
        GridAndConfig.__init__(self, config_file)

        config_file = os.path.join(rospkg.RosPack().get_path('torq_gcs'),
                                   'config', 'grid_only.rviz')

        self.frame.setMenuBar( None )
        self.frame.setStatusBar( None )
        self.frame.setHideButtonVisibility( False )

        reader = rviz.YamlConfigReader()
        config = rviz.Config()
        reader.readFile( config, config_file )
        self.frame.load( config )

        self.frame.setMenuBar( None )
        self.frame.setStatusBar( None )
        self.frame.setHideButtonVisibility( False )

def main():
    app = QApplication( sys.argv )

    config_file = os.path.join(rospkg.RosPack().get_path('torq_gcs'),
                               'config', 'race_conops.rviz')

    grid = GridAndConfig(config_file)

    layout = QVBoxLayout()
    layout.addWidget(grid.frame)

    grid.setLayout(layout)
    grid.resize( 2560, 1440 )

    # TODO(mereweth@jpl.nasa.gov) - pass gui object to config script

    grid.show()

    return app.exec_()

if __name__=="__main__":
    sys.exit(main())
