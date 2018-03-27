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

import numpy
np = numpy

import voxblox

from minsnap import utils

import transforms3d
from transforms3d import euler
from transforms3d import quaternions

import rospkg, rospy
from python_qt_binding import loadUi
from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *
from python_qt_binding.QtWidgets import *

from torq_gcs.fsp.ros_helpers import WorldMeshWorker, IntensityPCLWorker
from torq_gcs.fsp.ros_helpers import FreeSpheresWorker, PlaneControlWorker

class ESDFGUI(QWidget):
    def __init__(self,
                 global_dict,
                 node_name='voxblox_widgets',
                 new_node=False,
                 parent=None):
        super(ESDFGUI, self).__init__(parent)

        ui_file = os.path.join(rospkg.RosPack().get_path('torq_gcs'),
                               'resource', 'fsp', 'ESDF.ui')
        loadUi(ui_file, self)

        # NOTE (mereweth@jpl.nasa.gov) - this still does not bypass the GIL
        # unless rospy disables the GIL in C/C++ code
        self.ros_helper_thread = QThread()
        self.ros_helper_thread.app = self
        self.ros_helper_thread.start()

        if new_node:
            rospy.init_node(node_name, anonymous=True)

        self.world_mesh_worker = WorldMeshWorker(
                                            frame_id="local_origin",
                                            marker_topic="world_mesh_marker",
                                            refresh_ms=1000)

        self.esdf_distance_pcl_worker = IntensityPCLWorker(
                                            frame_id="local_origin",
                                            marker_topic="fsp_distance_pcl")

        self.tsdf_distance_pcl_worker = IntensityPCLWorker(
                                            frame_id="local_origin",
                                            marker_topic="tsdf_distance_pcl")

        self.tsdf_weight_pcl_worker = IntensityPCLWorker(
                                            frame_id="local_origin",
                                            marker_topic="tsdf_weight_pcl")

        self.free_spheres_worker = FreeSpheresWorker(
                                            frame_id="local_origin",
                                            marker_topic="free_spheres_marker")

        self.plane_control_worker = PlaneControlWorker(
                                            frame_id="local_origin",
                                            control_marker_topic="plane_control",
                                            plane_marker_topic="plane_marker")

        self.fill_spheres_worker = FreeSpheresWorker(
                                            frame_id="local_origin",
                                            marker_topic="fill_spheres_marker")

        self.num_pts = 100 * 1000

        self.free_sphere_radius = 0.1 # meters
        self.fill_sphere_radius = 0.2
        self.free_space_points = None
        self.fill_space_points = None
        self.tsdf_layer = None
        self.tsdf_map = None
        self.esdf_integrator = None

        # warn about overwriting?
        # if 'KEY' in global_dict.keys():
        global_dict['fsp_out_map'] = None
        #TODO(mereweth@jpl.nasa.gov) - custom signal for when update-function
        # is done? Is timing of signals guaranteed?
        global_dict['fsp_updated_signal_1'] = self.load_esdf_button.clicked
        global_dict['fsp_updated_signal_2'] = self.update_esdf_button.clicked
        self.global_dict = global_dict

        double_validator = QDoubleValidator(parent=self.free_sphere_radius_line_edit)
        double_validator.setBottom(0)
        self.free_sphere_radius_line_edit.setValidator(double_validator)
        self.free_sphere_radius_line_edit.setText(str(self.free_sphere_radius))
        self.free_sphere_radius_line_edit.textEdited.connect(self.on_free_sphere_radius_line_edit_text_edit)

        self.load_free_space_points_button.clicked.connect(self.on_load_free_space_points_button_click)
        self.clear_free_space_points_button.clicked.connect(self.on_clear_free_space_points_button_click)
        self.fill_free_space_points_button.clicked.connect(self.on_fill_free_space_points_button_click)

        self.update_esdf_button.clicked.connect(self.on_update_esdf_button_click)
        self.load_tsdf_button.clicked.connect(self.on_load_tsdf_button_click)
        self.load_esdf_button.clicked.connect(self.on_load_esdf_button_click)
        self.save_tsdf_button.clicked.connect(self.on_save_tsdf_button_click)
        self.save_esdf_button.clicked.connect(self.on_save_esdf_button_click)
        self.load_mesh_button.clicked.connect(self.on_load_mesh_button_click)
        self.load_filling_plane_button.clicked.connect(self.on_load_filling_plane_button_click)
        self.slice_value_spin_box.valueChanged.connect(self.on_slice_value_spin_box_value_changed)

    def on_free_sphere_radius_line_edit_text_edit(self, text):
        try:
            self.free_sphere_radius = float(text)
        except ValueError as e:
            self.free_sphere_radius_line_edit.text()
            self.free_sphere_radius_line_edit.setText(str(self.free_sphere_radius))

        if self.free_space_points is not None:
            self.free_spheres_worker.publish_spheres(
                        self.free_space_points['x'],
                        self.free_space_points['y'],
                        self.free_space_points['z'],
                        [self.free_sphere_radius] * len(self.free_space_points['x']))

    def on_update_esdf_button_click(self):
        if self.esdf_integrator is None:
            return

        # update and clear updated flag in TSDF voxels
        print("Overwriting any manual edits to ESDF")
        self.esdf_integrator.updateFromTsdfLayer(True)

    def on_load_esdf_button_click(self, checked=False, filename=None):
        if filename is None:
            filename = QFileDialog.getOpenFileName(self,
                                                   'Import ESDF', #path,
                                                   "ESDF proto files (*esdf.proto)")
            if filename and len(filename)>0:
                filename = filename[0]
            else:
                print("Invalid file path")
                return

        try:
            #TODO(mereweth@jpl.nasa.gov) - should we clear the TSDF?
            self.esdf_layer = voxblox.loadEsdfLayer(filename)
            self.global_dict['fsp_out_map'] = voxblox.EsdfMap(self.esdf_layer)
            self.slice_value_spin_box.setSingleStep(self.global_dict['fsp_out_map'].voxel_size)
        except Exception as e:
            print("Unknown error loading ESDF from {}".format(filename))
            print(e)
            return

    def on_load_tsdf_button_click(self, checked=False, filename=None):
        if filename is None:
            filename = QFileDialog.getOpenFileName(self,
                                                   'Import TSDF', #path,
                                                   "TSDF proto files (*tsdf.proto)")
            if filename and len(filename)>0:
                filename = filename[0]
            else:
                print("Invalid file path")
                return

        try:
            self.tsdf_layer = voxblox.loadTsdfLayer(filename)
            self.tsdf_map   = voxblox.TsdfMap(self.tsdf_layer)
            self.esdf_layer = voxblox.EsdfLayer(self.tsdf_layer.voxel_size,
                                                self.tsdf_layer.voxels_per_side)

            self.global_dict['fsp_out_map'] = voxblox.EsdfMap(self.esdf_layer)
            self.slice_value_spin_box.setSingleStep(self.global_dict['fsp_out_map'].voxel_size)

            # TODO(mereweth@jpl.nasa.gov) - specify config
            # how to deal with updating config?
            esdf_integrator_config = voxblox.EsdfIntegratorConfig()
            # TODO(mereweth@jpl.nasa.gov) - hacky; how to get truncation distance from NTSDF dump?
            esdf_integrator_config.min_distance_m = 0.22

            self.esdf_integrator = voxblox.EsdfIntegrator(
                                                esdf_integrator_config,
                                                self.tsdf_layer,
                                                self.esdf_layer)

        except Exception as e:
            print("Unknown error loading TSDF from {}".format(filename))
            print(e)
            return

    def on_save_esdf_button_click(self, checked=False, filename=None):
        if self.esdf_layer is None:
            return

        if filename is None:
            filename = QFileDialog.getSaveFileName(self,
                                                   'Save ESDF', #path,
                                                   "ESDF proto files (*esdf.proto)")
            if filename and len(filename)>0:
                filename = filename[0]
                #self.traj_file_label.setText( filename )
            else:
                print("Invalid file path")
                return

        try:
            print("Saving ESDF to {}".format(filename))
            self.esdf_layer.saveToFile(filename)
        except Exception as e:
            print("Could not save ESDF to {}".format(filename))
            print(e.message)

    def on_save_tsdf_button_click(self, checked=False, filename=None):
        if self.tsdf_layer is None:
            return

        if filename is None:
            filename = QFileDialog.getSaveFileName(self,
                                                   'Save TSDF', #path,
                                                   "TSDF proto files (*tsdf.proto)")
            if filename and len(filename)>0:
                filename = filename[0]
                #self.traj_file_label.setText( filename )
            else:
                print("Invalid file path")
                return

        try:
            print("Saving TSDF to {}".format(filename))
            self.tsdf_layer.saveToFile(filename)
        except Exception as e:
            print("Could not save TSDF to {}".format(filename))
            print(e.message)

    def on_load_free_space_points_button_click(self, checked=False, filename=None):
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
            self.free_space_points = utils.load_waypoints(filename)
            self.free_spheres_worker.publish_spheres(
                        self.free_space_points['x'],
                        self.free_space_points['y'],
                        self.free_space_points['z'],
                        [self.free_sphere_radius] * len(self.free_space_points['x']))
        except KeyError:
            print("Invalid file format")
            return
        except Exception as e:
            print("Unknown error loading waypoints from {}".format(filename))
            print(e)
            return

    def on_clear_free_space_points_button_click(self):
        if (self.free_space_points is None or self.esdf_layer is None):
            print("can not update")
            return

        #TODO(mereweth@jpl.nasa.gov) - batch process whatever is done here:
        for i in range(0, len(self.free_space_points['x'])):
            voxblox.clearSphereAroundPoint(
                                        self.esdf_layer,
                                        np.array([self.free_space_points['x'][i],
                                                  self.free_space_points['y'][i],
                                                  self.free_space_points['z'][i]],
                                                 dtype='double'),
                                        self.free_sphere_radius)

    def on_fill_free_space_points_button_click(self):


        # x_c = np.array([0.0,0,-0.5])
        # q = np.array([1.0,0,0,0])
        # x_s = np.array([8.0,0.6,2.0])
        if self.plane_control_worker.pos_global is None:
            print("No points to fill. Need to intialise plane first")
            return

        x_c = self.plane_control_worker.pos_global.copy()
        q = self.plane_control_worker.rot_global.copy()
        x_s = self.plane_control_worker.scale_global.copy()


        fill_points, r = self.create_points_from_plane(x_c, q, x_s)

        self.fill_space_points = dict()
        self.fill_space_points['x'] = fill_points[0,:]
        self.fill_space_points['y'] = fill_points[1,:]
        self.fill_space_points['z'] = fill_points[2,:]

        self.free_space_points = self.fill_space_points

        if (self.fill_space_points is None or self.esdf_layer is None):
            print("Nothing to fill")
            return

        #TODO(mereweth@jpl.nasa.gov) - batch process whatever is done here:
        for i in range(0, fill_points.shape[1]):
            voxblox.fillSphereAroundPoint(
                                        self.esdf_layer,
                                        fill_points[:,i],
                                        self.fill_sphere_radius)

    # def on_show_fill_spheres_click(self):
        self.fill_spheres_worker.publish_spheres(
                self.fill_space_points['x'],
                self.fill_space_points['y'],
                self.fill_space_points['z'],
                [r] * len(self.fill_space_points['x']))

    # def create_points_from_plane(self, x_c, q, x_s):
    #     """
    #     Generates a set of points in a mesh to fill a plane represented by a
    #     centre point, an orientation and a scale vector. A setting of r (radius)
    #     controls the spacing of the points in the mesh.
    #
    #     For use in plane filling applications
    #
    #     Args:
    #         x_c:    The centre location of the plane. x, y, z np array
    #         q:      The quaternion for the orientation of the plane (4x1 np arrray)
    #         x_s:    The scale for each axis (before transformation), in x, y, z as an np array
    #
    #     Out:
    #         points  A 3xN array for the points to represent the plane. Type double to be ready for tsdf integrator input
    #
    #     """
    #     # inputs
    #     R = transforms3d.quaternions.quat2mat(q) # Rotation matrix from quaternion input
    #
    #     # Settings
    #     r = self.fill_sphere_radius # Radius of spheres
    #
    #     if np.any(x_s<2*r):
    #         r = np.min(x_s)/2
    #
    #     # Create the meshgrid
    #     x_vec = np.linspace(-x_s[0]/2 + r,x_s[0]/2 - r,max(int(np.round((x_s[0]-2*r)/r,0)),1)+1)
    #     # y_vec = np.linspace(-x_s[1]/2 + r,x_s[1]/2 - r,max(int(np.round((x_s[1]-2*r)/r,0)),1)+1)
    #     y_vec = np.
    #     z_vec = np.linspace(-x_s[2]/2 + r,x_s[2]/2 - r,max(int(np.round((x_s[2]-2*r)/r,0)),1)+1)
    #
    #     XX, YY, ZZ = np.meshgrid(x_vec, y_vec, z_vec)
    #
    #     # FLatten into one set of points
    #     points = np.zeros([3,XX.size],dtype = 'double')
    #     points[0,:] = np.ndarray.flatten(XX)
    #     points[1,:] = np.ndarray.flatten(YY)
    #     points[2,:] = np.ndarray.flatten(ZZ)
    #
    #     # Rotate the points about the centre
    #     points = R.dot(points)
    #
    #     # Add the centre
    #     x_c.resize(3,1)
    #     points = points + x_c.dot(np.ones([1,XX.size]))
    #
    #     return points, r

    def create_points_from_plane(self, x_c, q, x_s):
        """
        Generates a set of points in a mesh to fill a plane represented by a
        centre point, an orientation and a scale vector. A setting of r (radius)
        controls the spacing of the points in the mesh.

        For use in plane filling applications

        Args:
            x_c:    The centre location of the plane. x, y, z np array
            q:      The quaternion for the orientation of the plane (4x1 np arrray)
            x_s:    The scale for each axis (before transformation), in x, y, z as an np array

        Out:
            points  A 3xN array for the points to represent the plane. Type double to be ready for tsdf integrator input

        """
        # inputs
        R = transforms3d.quaternions.quat2mat(q) # Rotation matrix from quaternion input

        # Settings
        r = self.fill_sphere_radius # Radius of spheres

        if np.any(x_s<2*r):
            r = np.min(x_s)/2

        # Create the meshgrid
        x_vec = np.linspace(-x_s[0]/2 + r,x_s[0]/2 - r,max(int(np.round((x_s[0]-2*r)/r,0)),1)+1)
        y_vec = np.linspace(-x_s[1]/2 + r,x_s[1]/2 - r,max(int(np.round((x_s[1]-2*r)/r,0)),1)+1)
        z_vec = np.linspace(-x_s[2]/2 + r,x_s[2]/2 - r,max(int(np.round((x_s[2]-2*r)/r,0)),1)+1)

        XX, YY, ZZ = np.meshgrid(x_vec, y_vec, z_vec)

        # FLatten into one set of points
        points = np.zeros([3,XX.size],dtype = 'double')
        points[0,:] = np.ndarray.flatten(XX)
        points[1,:] = np.ndarray.flatten(YY)
        points[2,:] = np.ndarray.flatten(ZZ)

        # Rotate the points about the centre
        points = R.dot(points)

        # Add the centre
        x_c.resize(3,1)
        points = points + x_c.dot(np.ones([1,XX.size]))

        return points, r

    def on_load_filling_plane_button_click(self):

        self.plane_control_worker.pos_global = np.array([0.,0,-1.0])
        self.plane_control_worker.rot_global = np.array([1.0,0,0,0])
        # self.plane_control_worker.scale_global = np.array([0.5,4.0,2.0])
        self.plane_control_worker.scale_global = np.array([self.fill_sphere_radius,4.0,2.0])

        self.plane_control_worker.make6DoFMarker()

        self.plane_control_worker.makeScaleMarkers()

        # 'commit' changes and send to all clients
        self.plane_control_worker.server.applyChanges()


    def on_load_mesh_button_click(self, checked=False, filename=None):
        if filename is None:
            filename = QFileDialog.getOpenFileName(self,
                           'Import mesh', #path,
                           "Polygon File Format(*.ply);; Collada (*.dae);; STL(*.stl)")
            if filename and len(filename)>0:
                filename = filename[0]
            else:
                print("Invalid file path")
                return

        try:
            self.world_mesh_worker.publish_marker("file://" + os.path.abspath(filename))
        except Exception as e:
            print("Unknown error loading mesh from {}".format(filename))
            print(e)
            return

    def on_slice_value_spin_box_value_changed(self):
        slice_value = self.slice_value_spin_box.value()

        if self.global_dict['fsp_out_map'] is not None:
            slice_xyzi = np.matrix(np.zeros((4, self.num_pts)))
            num_pts = self.num_pts
            try:
                # TODO(mereweth@jpl.nasa.gov) - buttons for xy, yz, xz slice
                num_pts = self.global_dict['fsp_out_map'].coordPlaneSliceGetDistance(2, # xy plane
                                                          slice_value,
                                                          slice_xyzi[0:3, :],
                                                          slice_xyzi[3, :].T,
                                                          self.num_pts)
            except RuntimeError as e:
                print(e)
                self.num_pts = 2*self.num_pts

            self.esdf_distance_pcl_worker.publish_point_cloud(slice_xyzi[0, 0:num_pts],
                                                          slice_xyzi[1, 0:num_pts],
                                                          slice_xyzi[2, 0:num_pts],
                                                          slice_xyzi[3, 0:num_pts])

        if self.tsdf_map is not None:
            # TODO(mereweth@jpl.nasa.gov) - use separate spin box for TSDF calculation
            slice_xyzi = np.matrix(np.zeros((5, self.num_pts)))
            try:
                # TODO(mereweth@jpl.nasa.gov) - buttons for xy, yz, xz slice
                num_pts = self.tsdf_map.coordPlaneSliceGetDistanceWeight(2, # xy plane
                                                          slice_value,
                                                          slice_xyzi[0:3, :],
                                                          slice_xyzi[3, :].T,
                                                          slice_xyzi[4, :].T,
                                                          self.num_pts)
            except RuntimeError as e:
                print(e)
                self.num_pts = 2*self.num_pts

            self.tsdf_distance_pcl_worker.publish_point_cloud(slice_xyzi[0, 0:num_pts],
                                                          slice_xyzi[1, 0:num_pts],
                                                          slice_xyzi[2, 0:num_pts],
                                                          slice_xyzi[3, 0:num_pts])

            self.tsdf_weight_pcl_worker.publish_point_cloud(slice_xyzi[0, 0:num_pts],
                                                          slice_xyzi[1, 0:num_pts],
                                                          slice_xyzi[2, 0:num_pts],
                                                          slice_xyzi[4, 0:num_pts])

def main():
    from torq_gcs.fsp.voxblox_widgets import ESDFGUI

    app = QApplication( sys.argv )

    global_dict = dict()
    fsp = ESDFGUI(global_dict, new_node=True)

    # TODO(mereweth@jpl.nasa.gov) - pass gui object to config script
    try:
        import imp
        conf = imp.load_source('torq_config',
                               os.path.expanduser('~/Desktop/environments/344.py'))
        conf.torq_config(fsp)
    except Exception as e:
        print("Error in config script")
        print(e)

    fsp.show()

    return app.exec_()

if __name__=="__main__":
    sys.exit(main())
