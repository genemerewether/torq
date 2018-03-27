# -*- coding: utf-8 -*-

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

import unittest
# from unittest.case import SkipTest

import sys, os
from python_qt_binding.QtWidgets import QApplication
#from python_qt_binding.QtTest import QTest

from torq_gcs.disc import line_simpl

class Base(unittest.TestCase):

    """
    Base class for tests

    This class defines a common `setUp` method that defines attributes which are used in the various tests.
    """

    def setUp(self):
        self.app = QApplication( sys.argv )

        global_dict = dict()
        self.rdp_gui = line_simpl.RDPGUI(global_dict)

        #self.rdp_gui.show()
        #return self.app.exec_()

    @classmethod
    def setUpClass(cls):
        """On inherited classes, run our `setUp` method"""
        # Inspired via http://stackoverflow.com/questions/1323455/python-unit-test-with-base-and-sub-class/17696807#17696807
        if cls is not Base and cls.setUp is not Base.setUp:
            orig_setUp = cls.setUp
            def setUpOverride(self, *args, **kwargs):
                Base.setUp(self)
                return orig_setUp(self, *args, **kwargs)
            cls.setUp = setUpOverride

    @classmethod
    def tearDownClass(cls):
        """Called once after all tests in this class."""
        pass

    def test_load_waypoints_missing_file(self):
        self.assertEqual(self.rdp_gui.in_waypoints, None)
        self.rdp_gui.on_load_waypoints_button_click(filename='/THIS_FILE_NO_EXIST')
        self.assertEqual(self.rdp_gui.in_waypoints, None)

class Input(Base):

    def setUp(self):
        pass

    def test_load_waypoints_figure_8(self):
        self.rdp_gui.on_load_waypoints_button_click(
                                                filename=os.path.abspath('../'))


if __name__ == '__main__':
    unittest.main()
