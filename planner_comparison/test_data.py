# source ~/catkin_ws/devel/setup.bash

import sys, os
import pickle

import numpy as np

import diffeo

""" 
Class to store results from batch tests with planners
"""
class DataStore:

    def __init__(self):

        # Each key in the dict is for one of the planners
        self.exit_flag = dict()
        self.comp_time = dict()
        self.traj_time = dict()
        self.min_distance = dict()
        self.mean_distance = dict()
        self.feasible = dict()
        self.poly = dict()
        self.waypoints_file = dict()
        self.esdf_file = dict()
        self.esdf_weight = dict()

        self.vel_max  = dict()#(vel_max)
        self.a_max  = dict()#(a_max)
        self.accel_average  = dict()#(accel_average)
        self.snap_average  = dict()#(snap_average)
        self.thr_max  = dict()#(thr_max)
        self.M_max  = dict()#(M_max)
        self.rpm_max  = dict()#(rpm_max)
        self.rpm_min  = dict()#(rpm_min)

        self.continuity = dict()

        # Initi arrays for each of the fields for each key
        for planner_type in ['unco','taco','astro','astro_cyl']:
            self.exit_flag[planner_type] = []
            self.comp_time[planner_type] = []
            self.traj_time[planner_type] = []
            self.min_distance[planner_type] = []
            self.mean_distance[planner_type] = []
            self.feasible[planner_type] = []
            self.poly[planner_type] = []
            self.waypoints_file[planner_type] = []
            self.esdf_file[planner_type] = []
            self.esdf_weight[planner_type] = []

            self.vel_max[planner_type] = [] #(vel_max)
            self.a_max[planner_type] = [] #(a_max)
            self.accel_average[planner_type] = [] #(accel_average)
            self.snap_average[planner_type] = [] #(snap_average)
            self.thr_max[planner_type] = [] #(thr_max)
            self.M_max[planner_type] = [] #(M_max)
            self.rpm_max[planner_type] = [] #(rpm_max)
            self.rpm_min[planner_type] = [] #(rpm_min)

            self.continuity[planner_type] = []

    def add_performance_data(self,planner_type,vel_max,a_max,accel_average,snap_average):#,thr_max,M_max,rpm_max,rpm_min):

        self.vel_max[planner_type].append(vel_max)
        self.a_max[planner_type].append(a_max)
        self.accel_average[planner_type].append(accel_average)
        self.snap_average[planner_type].append(snap_average)
        # self.thr_max[planner_type].append(thr_max)
        # self.M_max[planner_type].append(M_max)
        # self.rpm_max[planner_type].append(rpm_max)
        # self.rpm_min[planner_type].append(rpm_min)

    def get_performance_metrics_set(self,planner_type):

        n_samp = 100

        for i in range(len(self.poly[planner_type])):
            self.get_performance_metrics(n_samp,self.poly[planner_type][i])

    def get_performance_metrics(self,n_samp,poly=None):

        # Get poly
        if poly is None:
            return

        t_vec = np.linspace(poly['x'].x[0],poly['x'].x[-1],n_samp/2*poly['x'].x[-1])

        a_max = 0.0
        rpm_max = 0.0
        rpm_min = 999999990.0
        thr_max = 0.0
        vel_max = 0.0
        M_max = 0.0

        snap_sum = 0.0
        accel_sum = 0.0

        vel = np.array([poly['x'].derivative()(t_vec),
        poly['y'].derivative()(t_vec),
        poly['z'].derivative()(t_vec)])

        accel = np.array([poly['x'].derivative().derivative()(t_vec),
        poly['y'].derivative().derivative()(t_vec),
        poly['z'].derivative().derivative()(t_vec)])

        jerk = np.array([poly['x'].derivative().derivative().derivative()(t_vec),
        poly['y'].derivative().derivative().derivative()(t_vec),
        poly['z'].derivative().derivative().derivative()(t_vec)])

        snap = np.array([poly['x'].derivative().derivative().derivative().derivative()(t_vec),
        poly['y'].derivative().derivative().derivative().derivative()(t_vec),
        poly['z'].derivative().derivative().derivative().derivative()(t_vec)])

        yaw = poly['yaw'](t_vec)
        yaw_dot = poly['yaw'].derivative()(t_vec)
        yaw_ddot = poly['yaw'].derivative().derivative()(t_vec)

        for i in range(len(t_vec)):
            v_mag = np.linalg.norm(vel[:,i])
            a_mag = np.linalg.norm(accel[:,i])

            accel_sum += a_mag
            snap_sum += np.linalg.norm(snap[:,i])

            if a_mag > a_max:
                a_max = np.linalg.norm(accel)

            if v_mag > vel_max:
                vel_max = v_mag

        accel_average = accel_sum/len(t_vec)
        snap_average = snap_sum/len(t_vec)

        self.add_performance_data(planner_type,vel_max,a_max,accel_average,snap_average)#thr_max,M_max,rpm_max,rpm_min)

    """
    To check that the output piecewise polynomial is continuous at the breakpoints
    """
    def check_continuity(self,planner_type,eps=1e-6, rtol=1e-5, atol=1e-5):

        if not (hasattr(self,'continuity')):
            self.continuity = dict()
            self.continuity[planner_type] = []

        for j in range(len(self.poly[planner_type])):
            poly = self.poly[planner_type][j]
            # ppoly has no data before and after first transition points, so we
            # can't check those
            trans_times = poly['x'].x[1:-1]

            check_flag = True

            for key in ['x','y','z']:#] poly.keys():
                temp_ppoly = poly[key]
                if not np.allclose(temp_ppoly(trans_times - eps),temp_ppoly(trans_times + eps), rtol=rtol, atol=atol):
                    error = np.linalg.norm(temp_ppoly(trans_times - eps)-temp_ppoly(trans_times + eps))
                    print("Failed continuity check with error {} \n".format(error))
                    check_flag = False

                    break

            if check_flag:
                self.continuity[planner_type].append(True)
            else:
                self.continuity[planner_type].append(False)


    """
    Gets averages for the data across the runs of the planners for a given environment
    Outputs as row of data
    """
    def output_grouped_data(self,planner_type,delete_array=None):

        print("Planner is: {}".format(planner_type))

        """ Get the feasible results """
        feasible = np.array(data_out.feasible[planner_type])
        
        # Put each field into numpy arrays
        snap = np.array(data_out.snap_average[planner_type])
        comp_time = np.array(data_out.comp_time[planner_type])
        exit_flags = np.array(data_out.exit_flag[planner_type])
        traj_time = np.array(data_out.traj_time[planner_type])
        min_distance = np.array(data_out.min_distance[planner_type])
        if hasattr(self,'continuity'):
            continuity = np.array(self.continuity[planner_type])
        if planner_type is "astro":
            esdf_weight = np.array(data_out.esdf_weight[planner_type])

        print("Infeasible trajectories are: {}".format(np.where(1-feasible)))
        
        # index_list = np.array([0,2,7,9,12,29,44,46,48,49])
        # print("For {}, and traj:\n{}\nComp Time: \n{}\nMin Distance:\n{}\nfeasible:\n{}\nExit flags:\n{}\n".format(planner_type,index_list,comp_time[index_list],min_distance[index_list],feasible[index_list],exit_flags[index_list]))

        print("Comp time is:\n{}".format(comp_time))
        print("Traj time is:\n{}".format(traj_time))
        print("Delete array is:\n{}".format(delete_array))

        # Delete invalid iteams
        if delete_array is not None:
            # feasible[delete_array] = False
            # remove_num = np.size(delete_array)[0]
            feasible = np.delete(feasible,delete_array)
            snap = np.delete(snap,delete_array)
            comp_time = np.delete(comp_time,delete_array)
            exit_flags = np.delete(exit_flags,delete_array)
            traj_time = np.delete(traj_time,delete_array)
            min_distance = np.delete(min_distance,delete_array)
            continuity = np.delete(continuity,delete_array)

        remove_num = 0

        # Further checks for validity (snap and flags)
        if planner_type == "astro":
            print("ASTRO mutated in {} cases".format(np.sum(exit_flags == 2)))

            feasible[np.isnan(snap)] = False
            snap[np.isnan(snap)] = -1.0
            if np.size(snap) > 1:
                feasible[snap>10.0] = False
            #feasible[comp_time > 700.0] = False
            feasible[exit_flags==0] = False
        elif planner_type == "taco":
            if np.any(exit_flags==-2):

                print("Bad planned path for index {}".format(np.where(exit_flags == -2)[0]))
        elif planner_type is "astro_cyl":
            feasible[np.isnan(snap)] = False
            snap[np.isnan(snap)] = -1.0
            if np.size(snap) > 1:
                feasible[snap>10.0] = False
            feasible[exit_flags==0] = False

        feasible[continuity==False] = False

        print("Infeasible trajectories are: {}".format(np.where(1-feasible)))

        percent_feasible = float(np.sum(feasible))/float(np.size(feasible)-remove_num)*100.0

        print("\n\nPercent feasible for {} is {}%".format(planner_type,percent_feasible))
        delete_index = np.where(abs(np.array(feasible)-1))[0]
        

        """ Computation time """
        # comp_time[delete_index] = np.nan
        mean_comp = np.nanmean(comp_time)
        std_comp = np.sqrt(np.nanvar(comp_time))
        min_comp = np.nanmin(comp_time)
        max_comp = np.nanmax(comp_time)

        print("Computation time:\nAverage: {}\nMin: {}\nMax: {}\nSTEDV: {}\n\n".format(mean_comp,min_comp,max_comp,std_comp))

        """ Average Snap """
        snap[delete_index] = np.nan
        mean_snap = np.nanmean(snap)

        """ Average traj time """
        traj_time[delete_index] = np.nan
        mean_traj_time = np.nanmean(traj_time)

        """ Mean min distance """
        min_distance[delete_index] = np.nan
        mean_min_distance = np.nanmean(min_distance)

        """ Format data together """
        data = [planner_type,'opt_comp',mean_comp,min_comp,max_comp,std_comp,percent_feasible,mean_snap,mean_min_distance,mean_traj_time]

        return data


# Main part - write to CSV file

import csv


# Open file - change filename as desired (and add a filepath)
with open('198-B4_results_TEMP_V3.csv', 'wb') as csvfile:
# with open('344_results_v1_TEMP.csv', 'wb') as csvfile:
    fieldnames = ['Planner','Env.','Mean_comp','min_comp','max_comp','std_comp','perc_feas','mean_snap','mean_min_dist','mean_traj_time']
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(fieldnames)

    infeas_array = np.ones(100,dtype=bool)

    # Infeasible seed path for 198-B4 with 0.3 buffer
    infeas_array[88] = False

    # # Method to delete instances where there is bad data (old) - change below if needed)
    # use_check_data = False
    # if use_check_data:
    #   for planner_type, filename in zip(["unco","taco","astro","astro_cyl"],["unco_198-B4_V3_data.data","taco_198-B4_V3_data.data","astro_198-B4_V3_data.data","astro_cyl_198-B4_V3_data.data"]):
    #   # for planner_type, filename in zip(["unco","taco","astro","astro_cyl"],["unco_344_V1_data.data","taco_344_V1_data.data","astro_344_V1_new_data.data","astro_cyl_344_V1_data.data"]):

    #       if planner_type is not "taco":
    #           filename_nothing = filename[:-9]+"nothing_data.data"

    #           if planner_type is "astro_cyl":
    #               # filename_nothing = "astro_cyl_198-B4_V3_check_03_data.data"
    #               filename_nothing = "astro_cyl_344_V1_03_check_data.data"
    #               # filename_nothing = "astro_cyl_344_V1_03_CHECK_data.data"
    #               # filename_nothing = "astro_cyl_344_V1_03_CHECK_esdf_data.data"

    #           with open("Results/"+filename_nothing, 'rb') as f:
    #               data_nothing = pickle.load(f)

    #           # Track where it is infeaisble to start

    #           if planner_type is "astro_cyl":
    #               infeas_array = infeas_array*(np.array(data_nothing.exit_flag[planner_type])==-3)

    #           else:
    #               infeas_option = "distance"
    #               if infeas_option is "feasible_flag":
    #                   infeas_array = infeas_array*(1-np.array(data_nothing.feasible[planner_type]))
    #               elif infeas_option is "distance":
    #                   # infeasible if the distance is less than 0.1 (0.2 + 0.1 = 0.1 as the buffer used)
    #                   infeas_array = infeas_array*(np.array(data_nothing.min_distance[planner_type])<0.1)

    # Delete where it is feasible to start for both UNCO and ASTRO
    delete_array = np.where(1-infeas_array)[0]

    for planner_type, filename in zip(["unco","taco","astro","astro_cyl","astro_cyl"],["unco_198-B4_V3_03_data.data","taco_198-B4_V3_03_data.data","astro_198-B4_V3_03_data.data","astro_cyl_198-B4_V3_03_data.data","astro_cyl_198-B4_ESDF_V3_03_data.data"]):
    # for planner_type, filename in zip(["unco","taco","astro","astro_cyl","astro_cyl"],["unco_344_V1_03_data.data","taco_344_V1_03_data.data","astro_344_V1_03_data.data","astro_cyl_344_V1_03_new_data.data","astro_cyl_344_V1_03_esdf_new_data.data"]):
    
      # Saves into a reults folder (need to make it beforehand)
      with open("Results/"+filename, 'rb') as f:
          data_out = pickle.load(f)

      # Fill the data with the performance metrics
      data_out.get_performance_metrics_set(planner_type)

      # Check the continuity
      data_out.check_continuity(planner_type)

      # Get the grouped (average, variance) data
      data = data_out.output_grouped_data(planner_type,delete_array)

      # Write into the CSV
      writer.writerow(data)

   