# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example script to take into account intensity effects from impedance tables

:Authors: **Danilo Quartullo**
'''

from __future__ import division, print_function

import os
import sys

# stop the script 

from builtins import bytes, range, str

import matplotlib as mpl
import numpy as np

import matplotlib.pyplot as plt
import cupy as cp
 
from copy import deepcopy

from scipy.constants import c, e, m_p
from .drawnow import Waterfaller, WaterfallerOpt


import blond.utils.bmath as bm
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian,matched_from_distribution_function
from blond.beam.profile import CutOptions, Profile
from blond.impedances.impedance import (InducedVoltageFreq, InductiveImpedance,
                                        TotalInducedVoltage)
from blond.impedances.impedance_sources import InputTable
from blond.input_parameters.rf_parameters import RFStation, RFStationOptions, calculate_phi_s, calculate_Q_s
from blond.input_parameters.ring import Ring,RingOptions

from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
from blond.plots.plot_beams import plot_long_phase_space
from blond.plots.plot_impedance import (plot_impedance_vs_frequency,
                                        plot_induced_voltage_vs_bin_centers)
from blond.trackers.tracker import FullRingAndRF,RingAndRFTracker

# from numba import jit #NEED TO FIX THIS


# mpl.use('Agg')
def comp_phase_array(time_arr,phase_array,phase_n,n_times):
    """
    This function computes the phase array from the previous computed phase_array
    and adds the contribution of the current phase from n_time (in seconds) to 
    n_time + dt (in seconds) during the optimization run whose extent is given 
    by turn_range.

    phase_array has to be the phase computed upto n_time
    """

    index = np.where(time_arr <= n_times[0])[0][-1]
    index_i = np.where(time_arr <= n_times[1])[0][-1]

    # Generate a linear array from n_time - dt  to n_time 
    dummy_arr = np.linspace(phase_array[index_i], phase_n, index - index_i + 1, endpoint=True).reshape((index - index_i + 1,))

    dummy_sol = deepcopy(phase_array)

    dummy_sol[index_i:index+1] = dummy_arr
    dummy_sol[index+1:] = phase_n

    return dummy_sol

def run_simulation(turn_range,n_times,ring,phase_programme, Objects,sync_momentum, n_phase, final_run = False,init=False, voltages=None, full_phase = None):
    """
    This function runs the simulation for a given turn range and is made such that
    it allows the function to be passes to an optimizer that computes a constant phase
    at n_turn up to turn_range[1]. 

    Parameters: 
    turn_range: list of 3 integers
            | The turn range over which to simulate, the first two integers are the
            | start and end of the range, the third integer is the number of turns at 
            | which to compute the phase.

    n_times: list of 2 floats
            | The time at which the phase is computed (n_times[1]) and the 
            | time of the previous phase computation point (n_times[0])

    ring: Ring
            | The ring to simulate

    phase_programme: tuple of (time, phase) arrays
            | The phase programme to use
        
    Objects: list of tracking objects
            | The list of tracking objects to use
    
    sync_momentum: tuple of arrays
            | The sync momentum to use (time, momentum)
    
    n_phase: float
            | The phase to use at n_times[1] in the phase programme
     
    final_run: bool
            | If the function is called at the end of the optimization 
    
    init: bool
            | If the function is called at the beginning of the optimization 

    voltages: list of tuples (v1, v2)
            | The voltage programs to use (only needs to be imported in the first run,
            | after that it is saved in the RFStation object). 
            
    Returns:
    
    new_Objects: list of tracking objects 
            | The list of tracking objects saved at turn_range[2]
    
    phase_programme: tuple of (time, phase) arrays
            | The phase programme used
    """
    this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

    USE_GPU = True
    if USE_GPU:
        bm.use_py() # For the setup
    

    # SIMULATION PARAMETERS -------------------------------------------------------
    # Load momentum program?
    loading = True

    save_data = True
    name = 'opt_run'

    # Beam parameters
    n_particles = 0.9e13
    n_macroparticles = 1e6
    sigma_dt = 700e-9 /4  # [s]
    kin_beam_energy = 160e6  # [eV]

    # Machine and RF parameters
    radius = 25 # [m]
    bend_radius = 8.239 # [m]
    gamma_transition = 4.4  # [1]
    C = 2 * np.pi * radius  # [m]

    # Tracking details
    n_turns_between_two_plots = 10000 # One plot roughly every 10ms
    n_turns_between_wf_update = 200 # One waterfall update roughly every 0.2ms, 
    # NOTE: n_turns_between_wf_update also defines the dt of the reward computation

    # Derived parameters
    E_0 = m_p * c**2 / e    # [eV]
    tot_beam_energy = E_0 + kin_beam_energy  # [eV]
    momentum_compaction = 1 / gamma_transition**2  # [1]
    sync_ref_mom = np.sqrt(tot_beam_energy**2 - E_0**2)  # [eV/c]

    
    # # Cavities parameters
    # n_rf_systems = 2
    # # harmonic_numbers = 1
    # harmonic_numbers = [1,2]
    # voltage_program = [8e3, 2e3]  # [V]
    # phi_offset = np.pi

    # Cavities parameters
    n_rf_systems = 2
    # harmonic_numbers = 1
    harmonic_numbers = [1,2]
    voltage_program = [8e3, 5e3]  # [V]

    fmax=200e6
    width_bin = 1/(2*fmax)


    # CREATE THE ARRAYS FOR THE VOLTAGE AND PHASE OVER TIME----------------------
    if voltages == None:
        if not init:
            # NOTE: the loaded time series should have the same size as the synchotron momentum program 
            # If this is not the case, the program will crash

            v_p = np.load(this_directory + '../../v_programs_dt/'+ name +'.npy')
            v1 = v_p[0]
            v2 = v_p[1]
            time = v_p[2] 
            phase1 = np.ones_like(v1) * np.pi
            if not final_run:
                phase2 = comp_phase_array(sync_momentum[0],phase_programme,n_phase,n_times)

        else:
            v1 = voltage_program[0] * np.ones_like(sync_momentum[0])
            v2 = voltage_program[1] * np.ones_like(sync_momentum[0])
            if not final_run:
                phase2 = 1 * n_phase * np.ones_like(sync_momentum[0])

            phase1 = 1 * np.pi * np.ones_like(v1)

            # Allow for altering v1 and v2 using an interactive plot (only for the 1st opt. run)
            # plots = ProgramDefOpt(v1,v2, time=sync_momentum[0], sync_momentum=sync_momentum[1])
            
            # Compute the phase for the current optimization run
            # phase2 = comp_phase_array(sync_momentum[0],phase2,n_phase,n_time, dt)

        # if save_data:
            # plots.save_data('opt_run')

        #Construct tuples
        if loading:
            # Turn them back into seconds
            v1 = (sync_momentum[0], v1)
            v2 = (sync_momentum[0], v2)
            if final_run:
                phase2 = phase_programme
            else:
                phase2 = (sync_momentum[0], phase2)

            phase1 = (sync_momentum[0], phase1)
    
    else: 

        v1 = voltages[0]
        v2 = voltages[1]
        phase1 = np.ones_like(sync_momentum[0]) * np.pi
        if not init:
            phase2 = comp_phase_array(sync_momentum[0],phase_programme,n_phase,n_times)
        else: 
            phase2 = 1 * n_phase * np.ones_like(sync_momentum[0])
            
        phase2 = (sync_momentum[0], phase2)
        phase1 = (sync_momentum[0], phase1)
            

    # mpl.use('Agg')

    # # DEFINE REAL RING------------------------------------------------------------------
    if init or final_run: 

        

        n_slices = np.round(ring.t_rev[0] / width_bin).astype(int)

        RF_sct_par = RFStation(ring, harmonic_numbers,
                            (v1,v2), (phase1, phase2), n_rf_systems)

        my_beam = Beam(ring, n_macroparticles, n_particles)

        

        slice_beam = Profile(my_beam, CutOptions(cut_left=0,
                                                cut_right=ring.t_rev[0], n_slices=n_slices))
                # Finemet cavity
        F_C = np.loadtxt(this_directory +
                        '../../input_files/EX_02_Finemet.txt', dtype=float, skiprows=1)

        F_C[:, 3], F_C[:, 5], F_C[:, 7] = np.pi * F_C[:, 3] / \
            180, np.pi * F_C[:, 5] / 180, np.pi * F_C[:, 7] / 180

        option = "closed loop"

        if option == "open loop":
            Re_Z = F_C[:, 4] * np.cos(F_C[:, 3])
            Im_Z = F_C[:, 4] * np.sin(F_C[:, 3])
            F_C_table = InputTable(F_C[:, 0], 13 * Re_Z, 13 * Im_Z)
        elif option == "closed loop":
            Re_Z = F_C[:, 2] * np.cos(F_C[:, 5])
            Im_Z = F_C[:, 2] * np.sin(F_C[:, 5])
            F_C_table = InputTable(F_C[:, 0], 13 * Re_Z, 13 * Im_Z)
        elif option == "shorted":
            Re_Z = F_C[:, 6] * np.cos(F_C[:, 7])
            Im_Z = F_C[:, 6] * np.sin(F_C[:, 7])
            F_C_table = InputTable(F_C[:, 0], 13 * Re_Z, 13 * Im_Z)
        else:
            pass


        # steps
        steps = InductiveImpedance(my_beam, slice_beam, 34.6669349520904 / 10e9 *
                                ring.f_rev, RF_sct_par, deriv_mode='diff')
        # direct space charge
        dir_space_charge = InductiveImpedance(my_beam, slice_beam, -376.730313462
                                            / (ring.beta[0] * ring.gamma[0] ** 2),
                                            RF_sct_par)


        # TOTAL INDUCED VOLTAGE FROM IMPEDANCE------------------------------------------------

        imp_list = [F_C_table]

        ind_volt_freq = InducedVoltageFreq(my_beam, slice_beam, imp_list,
                                        frequency_resolution=2e5)


        total_induced_voltage = TotalInducedVoltage(my_beam, slice_beam,
                                                    [ind_volt_freq, steps, dir_space_charge])
        # Assemble the trackers

        ring_RF_section = RingAndRFTracker(RF_sct_par, my_beam, Profile=slice_beam, TotalInducedVoltage=total_induced_voltage)

        Full_rrf = FullRingAndRF([ring_RF_section])

    
        # DEFINE SLICES----------------------------------------------------------------
        
        # DEFINE BEAM------------------------------------------------------------------
        matched_from_distribution_function(my_beam, Full_rrf, distribution_type='parabolic_line',bunch_length=sigma_dt*4)

        my_beam.statistics()

        turn_saved = 1

        # LOAD IMPEDANCE TABLES--------------------------------------------------------

        # var = str(kin_beam_energy / 1e9)

        # # ejection kicker
        # Ekicker = np.loadtxt(this_directory + '../input_files/EX_02_Ekicker_1.4GeV.txt',
        #                     skiprows=1, dtype=complex,
        #                     converters={0: lambda s:
        #                                 complex(bytes(s).decode(
        #                                     'UTF-8').replace('i', 'j')),
        #                                 1: lambda s: complex(bytes(s).decode('UTF-8').replace('i', 'j'))})

        # Ekicker_table = InputTable(
        #     Ekicker[:, 0].real, Ekicker[:, 1].real, Ekicker[:, 1].imag)



    else: 
        ring_RF_section, slice_beam, total_induced_voltage, turn_saved = Objects[0], Objects[1], Objects[2] ,turn_range[2]

        # ring_RF_section.set_phase(ring, (phase1, phase2))
        

        ring_RF_section.rf_params.phi_rf_d = RFStationOptions().reshape_data((phase1, phase2),
                                                                            ring_RF_section.rf_params.n_turns,
                                                                            ring_RF_section.rf_params.n_rf,
                                                                            ring.cycle_time, 
                                                                            ring.RingOptions.t_start)
        
        ring_RF_section.rf_params.phi_rf = np.array(ring_RF_section.rf_params.phi_rf_d).astype(bm.precision.real_t)
        
        ring_RF_section.rf_params.phi_s = calculate_phi_s(ring_RF_section.rf_params, ring_RF_section.rf_params.Particle)
        ring_RF_section.rf_params.Q_s = calculate_Q_s(ring_RF_section.rf_params, ring_RF_section.rf_params.Particle)
        
        # # Maybe not even necessary
        # total_induced_voltage.profile = ring_RF_section.profile
        # total_induced_voltage.beam = ring_RF_section.beam


    # MONITOR----------------------------------------------------------------------

    # bunchmonitor = BunchMonitor(ring, RF_sct_par, my_beam,
    #                             this_directory + '../gpu_output_files/EX_02_output_data',
    #                             buffer_time=1)

    

    # PLOTS


    

    if USE_GPU:
        bm.use_gpu()
        # RF_sct_par.to_gpu()
        # my_beam.to_gpu()
        ring_RF_section.to_gpu()
        slice_beam.to_gpu()
        # Full_rrf.to_gpu()
        # ind_volt_freq.to_gpu()
        # steps.to_gpu()
        # dir_space_charge.to_gpu()
        total_induced_voltage.to_gpu()
        
        

        # map_ = [ring_RF_section] + [total_induced_voltage]#+ [slice_beam]   + [plots] #  + [bunchmonitor]

        # map_ = [ring_RF_section] + [total_induced_voltage] #+ [slice_beam] 

    # TRACKING + PLOTS-------------------------------------------------------------
    if final_run:
        format_options = {'dirname': this_directory +
                    '../gpu_output_files/EX_02_fig', 'linestyle': '-'}
        plots = Plot(ring, RF_sct_par, my_beam, n_turns_between_two_plots, ring.n_turns, 0,
                    ring.t_rev, - my_beam.sigma_dE *13, my_beam.sigma_dE * 13, xunit='s',
                    separatrix_plot=True, Profile=slice_beam, h5file=None,
                    histograms_plot=True, format_options=format_options, GPU_bool=USE_GPU)
        # THIS MADE ME 
        
        for i in range(1, ring.n_turns + 1):
            
            ring_RF_section.track()
            ring_RF_section.totalInducedVoltage.track()
            ring_RF_section.profile.track()
            plots.track()
            
            # if i==turn_range[0]: 
            #     # We want to save the initial beam profile
            #     wf = Waterfaller(ring_RF_section.profile, i,  sampling_d_turns = n_turns_between_wf_update , 
            #                     n_turns= turn_range[1], # Set to None if you want to plot individual ranges
            #                     traces = None, USE_GPU=USE_GPU)

            # Update the waterfall plot
            # if not init:
            #     wf.update(ring_RF_section.profile,i
                
                # Plots
            if (i % n_turns_between_two_plots) == 0:
                print('We are on turn ', i , '= ', ring.cycle_time[i]*1000 , ' ms')
                slice_beam.beam_spectrum_freq_generation(slice_beam.n_slices)
                slice_beam.beam_spectrum_generation(slice_beam.n_slices)

                    # if USE_GPU:
                    #     bm.use_cpu()
                    #     # total_induced_voltage.to_cpu()
                    #     # ind_volt_freq.to_cpu()
                    #     slice_beam.to_cpu()

                    # # plot_impedance_vs_frequency(ind_volt_freq, figure_index=i, cut_up_down=(0, 1000), cut_left_right=(0, 3e9),
                    # #                             show_plots=False,
                    # #                             plot_total_impedance=False, style='-', plot_interpolated_impedances=False,
                    # #                             plot_spectrum=False, dirname=this_directory + '../gpu_output_files/EX_02_fig', log=True)

                    # # plot_induced_voltage_vs_bin_centers(total_induced_voltage, style='.',
                    # #                                     dirname=this_directory + '../gpu_output_files/EX_02_fig', show_plots=False)

                    # if USE_GPU:
                    #     bm.use_gpu()
                    #     # total_induced_voltage.to_gpu()
                    #     # ind_volt_freq.to_gpu()
                    #     slice_beam.to_gpu()

        if USE_GPU:
            bm.use_cpu()
            # RF_sct_par.to_cpu()
            # my_beam.to_cpu()
            ring_RF_section.to_cpu()
            slice_beam.to_cpu()
            # Full_rrf.to_cpu()
            # ind_volt_freq.to_cpu()
            # steps.to_cpu()
            # dir_space_charge.to_cpu()
            total_induced_voltage.to_cpu()
        
        return
    
    else:

        for i in range(turn_range[0], turn_range[1] + 1):

            if i == turn_saved:
                if USE_GPU:
                    ring_RF_section.to_cpu()
                    slice_beam.to_cpu()
                    total_induced_voltage.to_cpu()
                    new_Objects = [deepcopy(ring_RF_section), deepcopy(slice_beam), deepcopy(total_induced_voltage)]
                    ring_RF_section.to_gpu()
                    slice_beam.to_gpu()
                    total_induced_voltage.to_gpu()
                else:
                    new_Objects = [deepcopy(ring_RF_section), deepcopy(slice_beam), deepcopy(total_induced_voltage)]

            
            ring_RF_section.track()
            ring_RF_section.totalInducedVoltage.track()
            ring_RF_section.profile.track()

            
            if final_run:
                plots.track()
            if i==turn_range[0]: 
                # We want to save the initial beam profile
                
                wf = Waterfaller(ring_RF_section.profile, i,  sampling_d_turns = n_turns_between_wf_update , 
                                n_turns= turn_range[1], # Set to None if you want to plot individual ranges
                                traces = None, USE_GPU=USE_GPU)

            # Update the waterfall plot
            if not init:
                wf.update(ring_RF_section.profile,i)

            
                
            # if final_run: 
            #     # Plots
            #     if (i % n_turns_between_two_plots) == 0:
            #         print('We are on turn ', i , '= ', ring.cycle_time[i]*1000 , ' ms')
            #         slice_beam.beam_spectrum_freq_generation(slice_beam.n_slices)
            #         slice_beam.beam_spectrum_generation(slice_beam.n_slices)

                    # if USE_GPU:
                    #     bm.use_cpu()
                    #     # total_induced_voltage.to_cpu()
                    #     # ind_volt_freq.to_cpu()
                    #     slice_beam.to_cpu()

                    # # plot_impedance_vs_frequency(ind_volt_freq, figure_index=i, cut_up_down=(0, 1000), cut_left_right=(0, 3e9),
                    # #                             show_plots=False,
                    # #                             plot_total_impedance=False, style='-', plot_interpolated_impedances=False,
                    # #                             plot_spectrum=False, dirname=this_directory + '../gpu_output_files/EX_02_fig', log=True)

                    # # plot_induced_voltage_vs_bin_centers(total_induced_voltage, style='.',
                    # #                                     dirname=this_directory + '../gpu_output_files/EX_02_fig', show_plots=False)

                    # if USE_GPU:
                    #     bm.use_gpu()
                    #     # total_induced_voltage.to_gpu()
                    #     # ind_volt_freq.to_gpu()
                    #     slice_beam.to_gpu()

        if USE_GPU:
            bm.use_cpu()
            # RF_sct_par.to_cpu()
            # my_beam.to_cpu()
            ring_RF_section.to_cpu()
            slice_beam.to_cpu()
            # Full_rrf.to_cpu()
            # ind_volt_freq.to_cpu()
            # steps.to_cpu()
            # dir_space_charge.to_cpu()
            total_induced_voltage.to_cpu()

        return new_Objects, phase2

def run_simulation_int(turn_range,n_times,ring,phase_programme, Objects,sync_momentum, n_phase, final_run = False,init=False, unique = False, voltages = None):
    """
    This function runs the simulation for a given turn range and is made such that
    it allows the function to be passes to an optimizer that computes a constant phase
    at n_turn up to turn_range[1]. 

    Parameters: 
    turn_range: list of 3 integers
            | The turn range over which to simulate, the first two integers are the
            | start and end of the range, the third integer is the number of turns at 
            | which to compute the phase.

    n_times: list of 2 floats
            | The time at which the phase is computed (n_times[1]) and the 
            | time of the previous phase computation point (n_times[0])

    ring: Ring
            | The ring to simulate

    phase_programme: tuple of (time, phase) arrays
            | The phase programme to use
        
    Objects: list of tracking objects
            | The list of tracking objects to use
    
    sync_momentum: tuple of arrays
            | The sync momentum to use (time, momentum)
    
    n_phase: float
            | The phase to use at n_time in the phase programme
    
    final_run: bool
            | If the function is called at the end of the optimization 
    
    init: bool
            | If the function is called at the beginning of the optimization 

    unique: bool
            | If only one trace is generated at n_time, if False then the optimizer
            | will generate multiple traces and the average error over the entire 
            | turn range will be returned
            

    """
    this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

    USE_GPU = True
    if USE_GPU:
        bm.use_py() # For the setup

    # SIMULATION PARAMETERS -------------------------------------------------------
    # Load momentum program?
    loading = True

    # Option to use loaded programme or define it right now
    # if init:
    #     sequential = False
    # else:
    #     sequential = True

    # save_data = True
    name = 'opt_run'

    # Beam parameters
    n_particles = 0.9e13
    n_macroparticles = 1e6
    sigma_dt = 700e-9 /4  # [s]

    # Machine and RF parameters

    # Tracking details
    n_turns_between_two_plots = 10000 # One plot roughly every 3ms
    n_turns_between_wf_update = 200 # One plot roughly every 0.1ms, 
    # NOTE: n_turns_between_wf_update also defines the dt of the reward computation

    # Derived parameters

    
    # # Cavities parameters
    # n_rf_systems = 2
    # # harmonic_numbers = 1
    # harmonic_numbers = [1,2]
    # voltage_program = [8e3, 2e3]  # [V]
    # phi_offset = np.pi

    # Cavities parameters
    n_rf_systems = 2
    # harmonic_numbers = 1
    harmonic_numbers = [1,2]
    voltage_program = [8e3, 5e3]  # [V]

    fmax=200e6
    width_bin = 1/(2*fmax)


    # CREATE THE ARRAYS FOR THE VOLTAGE AND PHASE OVER TIME----------------------
    # CREATE THE ARRAYS FOR THE VOLTAGE AND PHASE OVER TIME----------------------
    if voltages == None:
        if not init:
            # NOTE: the loaded time series should have the same size as the synchotron momentum program 
            # If this is not the case, the program will crash

            v_p = np.load(this_directory + '../../v_programs_dt/'+ name +'.npy')
            v1 = v_p[0]
            v2 = v_p[1]
            time = v_p[2] 
            phase1 = np.ones_like(v1) * np.pi
            if not final_run:
                phase2 = comp_phase_array(sync_momentum[0],phase_programme,n_phase,n_times)

        else:
            v1 = voltage_program[0] * np.ones_like(sync_momentum[0])
            v2 = voltage_program[1] * np.ones_like(sync_momentum[0])
            if not final_run:
                phase2 = 1 * n_phase * np.ones_like(sync_momentum[0])

            phase1 = 1 * np.pi * np.ones_like(v1)

            # Allow for altering v1 and v2 using an interactive plot (only for the 1st opt. run)
            # plots = ProgramDefOpt(v1,v2, time=sync_momentum[0], sync_momentum=sync_momentum[1])
            
            # Compute the phase for the current optimization run
            # phase2 = comp_phase_array(sync_momentum[0],phase2,n_phase,n_time, dt)

        # if save_data:
            # plots.save_data('opt_run')

        #Construct tuples
        if loading:
            # Turn them back into seconds
            v1 = (sync_momentum[0], v1)
            v2 = (sync_momentum[0], v2)
            if final_run:
                phase2 = phase_programme
            else:
                phase2 = (sync_momentum[0], phase2)

            phase1 = (sync_momentum[0], phase1)
    
    else: 

        v1 = voltages[0]
        v2 = voltages[1]
        phase1 = np.ones_like(sync_momentum[0]) * np.pi
        if not init:
            phase2 = comp_phase_array(sync_momentum[0],phase_programme,n_phase,n_times)
        else: 
            phase2 = 1 * n_phase * np.ones_like(sync_momentum[0])
            
        phase2 = (sync_momentum[0], phase2)
        phase1 = (sync_momentum[0], phase1)
            

    # mpl.use('Agg')


    # if not init:
    #     # NOTE: the loaded time series should have the same size as the synchotron momentum program 
    #     # If this is not the case, the program will crash

    #     # v_p = np.load(this_directory + '../../v_programs_dt/'+ name +'.npy')
    #     # v1 = v_p[0]
    #     # v2 = v_p[1]
    #     phase1 = np.ones_like(sync_momentum[0]) * np.pi
    #     phase2 = comp_phase_array(sync_momentum[0],phase_programme,n_phase,n_times)

    # else:
    #     v1 = voltage_program[0] * np.ones_like(sync_momentum[0])
    #     v2 = voltage_program[1] * np.ones_like(sync_momentum[0])
    #     phase2 = 1 * n_phase * np.ones_like(sync_momentum[0])
    #     phase1 = 1 * np.pi * np.ones_like(v1)
    #     n_turns_between_wf_update = 20

    #     # Allow for altering v1 and v2 using an interactive plot (only for the 1st opt. run)
    #     # plots = ProgramDefOpt(v1,v2, time=sync_momentum[0], sync_momentum=sync_momentum[1])
        
    #     # Compute the phase for the current optimization run
    #     # phase2 = comp_phase_array(sync_momentum[0],phase2,n_phase,n_time, dt)

    # # if save_data:
    # #     plots.save_data('opt_run')

    # #Construct tuples
    # if loading:
    #     # Turn them back into seconds
    #     # v1 = (sync_momentum[0], v1)
    #     # v2 = (sync_momentum[0], v2)
    #     phase2 = (sync_momentum[0], phase2)
    #     phase1 = (sync_momentum[0], phase1)
        

    # # mpl.use('Agg')

    # # DEFINE REAL RING------------------------------------------------------------------
    if init: 
        turn_saved = 1
        n_slices = np.round(ring.t_rev[0] / width_bin).astype(int)

        RF_sct_par = RFStation(ring, harmonic_numbers,
                            (v1,v2), (phase1, phase2), n_rf_systems)

        my_beam = Beam(ring, n_macroparticles, n_particles)

        slice_beam = Profile(my_beam, CutOptions(cut_left=0,
                                                cut_right=ring.t_rev[0], n_slices=n_slices))

                # Finemet cavity
        F_C = np.loadtxt(this_directory +
                        '../../input_files/EX_02_Finemet.txt', dtype=float, skiprows=1)

        F_C[:, 3], F_C[:, 5], F_C[:, 7] = np.pi * F_C[:, 3] / \
            180, np.pi * F_C[:, 5] / 180, np.pi * F_C[:, 7] / 180

        option = "closed loop"

        if option == "open loop":
            Re_Z = F_C[:, 4] * np.cos(F_C[:, 3])
            Im_Z = F_C[:, 4] * np.sin(F_C[:, 3])
            F_C_table = InputTable(F_C[:, 0], 13 * Re_Z, 13 * Im_Z)
        elif option == "closed loop":
            Re_Z = F_C[:, 2] * np.cos(F_C[:, 5])
            Im_Z = F_C[:, 2] * np.sin(F_C[:, 5])
            F_C_table = InputTable(F_C[:, 0], 13 * Re_Z, 13 * Im_Z)
        elif option == "shorted":
            Re_Z = F_C[:, 6] * np.cos(F_C[:, 7])
            Im_Z = F_C[:, 6] * np.sin(F_C[:, 7])
            F_C_table = InputTable(F_C[:, 0], 13 * Re_Z, 13 * Im_Z)
        else:
            pass


        # steps
        steps = InductiveImpedance(my_beam, slice_beam, 34.6669349520904 / 10e9 *
                                ring.f_rev, RF_sct_par, deriv_mode='diff')
        # direct space charge
        dir_space_charge = InductiveImpedance(my_beam, slice_beam, -376.730313462
                                            / (ring.beta[0] * ring.gamma[0] ** 2),
                                            RF_sct_par)


        # TOTAL INDUCED VOLTAGE FROM IMPEDANCE------------------------------------------------

        imp_list = [F_C_table]

        ind_volt_freq = InducedVoltageFreq(my_beam, slice_beam, imp_list,
                                        frequency_resolution=2e5)


        total_induced_voltage = TotalInducedVoltage(my_beam, slice_beam,
                                                    [ind_volt_freq, steps, dir_space_charge])


        # DEFINE RING AND RF TRACKER--------------------------------------------------
        ring_RF_section = RingAndRFTracker(RF_sct_par, my_beam, Profile=slice_beam, TotalInducedVoltage=total_induced_voltage)

        Full_rrf = FullRingAndRF([ring_RF_section])

    
        # DEFINE SLICES----------------------------------------------------------------
        
        # DEFINE BEAM------------------------------------------------------------------
        matched_from_distribution_function(my_beam, Full_rrf, distribution_type='parabolic_line',bunch_length=sigma_dt*4)

        # LOAD IMPEDANCE TABLES--------------------------------------------------------

        # var = str(kin_beam_energy / 1e9)

        # # ejection kicker
        # Ekicker = np.loadtxt(this_directory + '../input_files/EX_02_Ekicker_1.4GeV.txt',
        #                     skiprows=1, dtype=complex,
        #                     converters={0: lambda s:
        #                                 complex(bytes(s).decode(
        #                                     'UTF-8').replace('i', 'j')),
        #                                 1: lambda s: complex(bytes(s).decode('UTF-8').replace('i', 'j'))})

        # Ekicker_table = InputTable(
        #     Ekicker[:, 0].real, Ekicker[:, 1].real, Ekicker[:, 1].imag)

    else: 
        ring_RF_section, slice_beam, total_induced_voltage, turn_saved = deepcopy(Objects[0]), deepcopy(Objects[1]), deepcopy(Objects[2]),  turn_range[2]

        ring_RF_section.rf_params.phi_rf_d = RFStationOptions().reshape_data((phase1, phase2),
                                                                            ring_RF_section.rf_params.n_turns,
                                                                            ring_RF_section.rf_params.n_rf,
                                                                            ring.cycle_time, 
                                                                            ring.RingOptions.t_start)
        
        ring_RF_section.rf_params.phi_rf = np.array(ring_RF_section.rf_params.phi_rf_d).astype(bm.precision.real_t)
        
        ring_RF_section.rf_params.phi_s = calculate_phi_s(ring_RF_section.rf_params, ring_RF_section.rf_params.Particle)
        ring_RF_section.rf_params.Q_s = calculate_Q_s(ring_RF_section.rf_params, ring_RF_section.rf_params.Particle)
        

        # total_induced_voltage.profile = ring_RF_section.profile
        # total_induced_voltage.beam = ring_RF_section.beam
        # total_induced_voltage.profile = ring_RF_section.profile
        # plt.plot(old_phi2, label='phi2_old')
        # plt.plot(ring_RF_section.rf_params.phi_rf_d[1], label='phi2')
        # plt.legend()
        # plt.show()
        

    # DEFINE REFERENCE BEAM------------------------------------------------------------------
    # DEFINE REFERENCE RING + BEAM --------------------------------------------------

    # ref_ring = Ring(C, momentum_compaction, sync_ref_mom,
    #                 Proton())

    # ref_beam = Beam(ref_ring, n_macroparticles, n_particles)

    # ref_RF_section = RFStation(ref_ring, harmonic_numbers, [voltage_program[0], voltage_program[1]], [np.pi, np.pi], n_rf_systems)


    # ref_ring_RF_section = RingAndRFTracker(ref_RF_section, ref_beam)

    # Full_ref_rrf = FullRingAndRF([ref_ring_RF_section])
    # # DEFINE REFERENCE SLICES----------------------------------------------------------------
    # ref_slice_beam = Profile(ref_beam, CutOptions(cut_left=0,
    #                                          cut_right=ring.t_rev[0], n_slices=n_slices))
    # matched_from_distribution_function(ref_beam, Full_ref_rrf, distribution_type='parabolic_line',bunch_length=sigma_dt*4,process_pot_well = False)


    # PREPARE REFERENCE BEAM PROFILE FOR OPTIMIZER-----------------------------------

    # MONITOR----------------------------------------------------------------------

    # bunchmonitor = BunchMonitor(ring, RF_sct_par, my_beam,
    #                             this_directory + '../gpu_output_files/EX_02_output_data',
    #                             buffer_time=1)

    


    
    # PLOTS


    # plots = Plotgpu(ring, RF_sct_par, my_beam, 1, n_turns, 0,
    #              5.72984173562e-7, - my_beam.sigma_dE * 4.2, my_beam.sigma_dE * 4.2, xunit='s',
    #              separatrix_plot=True, Profile=slice_beam, h5file=this_directory + '../output_files/EX_02_output_data',
    #              histograms_plot=True, format_options=format_options)


    

    if USE_GPU:
        bm.use_gpu()
        # RF_sct_par.to_gpu()
        # my_beam.to_gpu()
        ring_RF_section.to_gpu()
        # slice_beam.to_gpu()
        # Full_rrf.to_gpu()
        # ind_volt_freq.to_gpu()
        # steps.to_gpu()
        # dir_space_charge.to_gpu()
        # total_induced_voltage.to_gpu()
        # total_induced_voltage.profile = ring_RF_section.profile
        # total_induced_voltage.beam = ring_RF_section.beam
        
        
    # Plotgpu has to go after sending data to GPU, otherwise, another fix is required as well


    
    map_ = [ring_RF_section] #+ [slice_beam]

    # TRACKING + PLOTS-------------------------------------------------------------
    if not unique:    
        for i in range(turn_range[0], turn_range[1] + 1):


            # assert ring_RF_section.counter[0]+1 <= turn_range[1] and ring_RF_section.counter[0]+1 >= turn_range[0] , "Turn counter {x} out of bounds {y}".format(x=ring_RF_section.counter[0]+1, y=turn_range)

            ring_RF_section.track()
            ring_RF_section.totalInducedVoltage.track()
            ring_RF_section.profile.track()

            # if not init: 
            #     print(ring_RF_section.beam.dt[10])

            
            if i==turn_range[0]: 
                
                wf = WaterfallerOpt(ring_RF_section.profile, i,  sampling_d_turns = n_turns_between_wf_update , 
                                n_turns= turn_range[1], # Set to None if you want to plot individual ranges
                                traces = None, USE_GPU=USE_GPU)

            # Update the waterfall plot
            # if i == turn_range[1]:
            profiles = wf.update(ring_RF_section.profile,i)
            # else:
            #     wf.update(ring_RF_section.profile,i)
            

            
            if final_run: 
                # Plots
                if (i % n_turns_between_two_plots) == 0:
                    print('We are on turn ', i , '= ', ring.cycle_time[i]*1000 , ' ms')
                    slice_beam.beam_spectrum_freq_generation(slice_beam.n_slices)
                    slice_beam.beam_spectrum_generation(slice_beam.n_slices)

                    # if USE_GPU:
                    #     bm.use_cpu()
                    #     # total_induced_voltage.to_cpu()
                    #     # ind_volt_freq.to_cpu()
                    #     slice_beam.to_cpu()

                    # # plot_impedance_vs_frequency(ind_volt_freq, figure_index=i, cut_up_down=(0, 1000), cut_left_right=(0, 3e9),
                    # #                             show_plots=False,
                    # #                             plot_total_impedance=False, style='-', plot_interpolated_impedances=False,
                    # #                             plot_spectrum=False, dirname=this_directory + '../gpu_output_files/EX_02_fig', log=True)

                    # # plot_induced_voltage_vs_bin_centers(total_induced_voltage, style='.',
                    # #                                     dirname=this_directory + '../gpu_output_files/EX_02_fig', show_plots=False)

                    # if USE_GPU:
                    #     bm.use_gpu()
                    #     # total_induced_voltage.to_gpu()
                    #     # ind_volt_freq.to_gpu()
                    #     slice_beam.to_gpu()

    else:
        for i in range(turn_range[0], turn_range[1] + 1):


            # assert ring_RF_section.counter[0]+1 <= turn_range[1] and ring_RF_section.counter[0]+1 >= turn_range[0] , "Turn counter {x} out of bounds {y}".format(x=ring_RF_section.counter[0]+1, y=turn_range)

            ring_RF_section.track()
            ring_RF_section.totalInducedVoltage.track()
            ring_RF_section.profile.track()


            # if not init: 
            #     print(ring_RF_section.beam.dt[10])
            if i==turn_range[0]: 
                # We want to save the initial beam profile
                
                wf = WaterfallerOpt(ring_RF_section.profile, i,  sampling_d_turns = n_turns_between_wf_update , 
                                n_turns= turn_range[1], # Set to None if you want to plot individual ranges
                                traces = None, USE_GPU=USE_GPU)
            
            if i==turn_saved:
                if USE_GPU:
                    profiles = [[deepcopy(ring_RF_section.profile.bin_centers.get()), deepcopy(ring_RF_section.profile.n_macroparticles.get())]]
                else:
                    profiles = [[deepcopy(ring_RF_section.profile.bin_centers), deepcopy(ring_RF_section.profile.n_macroparticles)]]
            
            wf.update(ring_RF_section.profile,i)
            
            if final_run: 
                # Plots
                if (i % n_turns_between_two_plots) == 0:
                    print('We are on turn ', i , '= ', ring.cycle_time[i]*1000 , ' ms')
                    slice_beam.beam_spectrum_freq_generation(slice_beam.n_slices)
                    slice_beam.beam_spectrum_generation(slice_beam.n_slices)

                    # if USE_GPU:
                    #     bm.use_cpu()
                    #     # total_induced_voltage.to_cpu()
                    #     # ind_volt_freq.to_cpu()
                    #     slice_beam.to_cpu()

                    # # plot_impedance_vs_frequency(ind_volt_freq, figure_index=i, cut_up_down=(0, 1000), cut_left_right=(0, 3e9),
                    # #                             show_plots=False,
                    # #                             plot_total_impedance=False, style='-', plot_interpolated_impedances=False,
                    # #                             plot_spectrum=False, dirname=this_directory + '../gpu_output_files/EX_02_fig', log=True)

                    # # plot_induced_voltage_vs_bin_centers(total_induced_voltage, style='.',
                    # #                                     dirname=this_directory + '../gpu_output_files/EX_02_fig', show_plots=False)

                    # if USE_GPU:
                    #     bm.use_gpu()
                    #     # total_induced_voltage.to_gpu()
                    #     # ind_volt_freq.to_gpu()
                    #     slice_beam.to_gpu()

    if USE_GPU:
        bm.use_cpu()
        # RF_sct_par.to_cpu()
        # my_beam.to_cpu()
        ring_RF_section.to_cpu()

        slice_beam.to_cpu()
        # Full_rrf.to_cpu()
        # ind_volt_freq.to_cpu()
        # steps.to_cpu()
        # dir_space_charge.to_cpu()
        total_induced_voltage.to_cpu()

    
    return profiles

if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('TkAgg')
    bm.use_precision('single')

    #Read the phase profile from the best fit
    this_directory = os.path.dirname(os.path.abspath(__file__))
    phase = np.load(this_directory + '/../../Sols/shape_full_58_cumsum/phase_array.npy') # (time, phase2)
    sync_momentum = np.load(this_directory + '/../../input_files/1_4GeVOperational.npy')# in eV/c 

    t_arr = sync_momentum[0] #in s
    program = sync_momentum[1] * 1e-4 # in Tesla
    # Machine and RF parameters
    radius = 25 # [m]
    bend_radius = 8.239 # [m]
    gamma_transition = 4.4  # [1]
    C = 2 * np.pi * radius  # [m]
    momentum_compaction = 1 / (gamma_transition)**2 # [-]

    sync_momentum = program * bend_radius *c #/1e9 # in GeV/c
    inj_t = 275  # in s
    ext_t = 805  # in s

    index_inj = np.where(t_arr <= inj_t)[0][-1]
    index_ext = np.where(t_arr >= ext_t)[0][0]

    t_arr = t_arr[index_inj:index_ext]
    t_arr = (t_arr - t_arr[0])/1e3
    sync_momentum = sync_momentum[index_inj:index_ext]


    t_range = [inj_t, ext_t]
    total_sim_t = (ext_t - inj_t)/1000
    t_range = [0,t_arr[-1]]
    dt = 0.1 # in ms for the timeseries
    sync_momentum = (t_arr, sync_momentum)

    phase = (phase[0][0], phase[0][1])

    ring = Ring(C, momentum_compaction, sync_momentum,
            Proton())

    # Run the simulation
    run_simulation(None,None,ring,phase, None,sync_momentum, None, final_run = True,init=True, voltages=None)

