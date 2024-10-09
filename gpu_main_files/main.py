import numpy as np 
from scipy.constants import c, e, m_p
from scipy.signal import butter,lfilter, savgol_filter, find_peaks
import os 
import matplotlib.pyplot as plt
import os 
from functools import partial
# import sys 
from skopt import gp_minimize 
from skopt.plots import plot_gaussian_process 

# Home-made functions and classes

from utils_opt.func_utils import *
# from scipy.optimize import minimize_scalar
from utils_opt.drawnow import ProgramDefOpt, tolSetter


#BLonD imports
import blond.utils.bmath as bm
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import matched_from_distribution_function
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.trackers.tracker import FullRingAndRF,RingAndRFTracker



def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def range_comp(v1, ring):
    '''
    Computes the points in time at which the 
    phase program is computed at such that points in time
    in which the stable phase (first order approximation)
    has a discontinuity or if its derivative is discontinuous
    (the rest is done by interpolation)
    '''
    

    stable_phase = -np.arcsin(np.array(ring.delta_E[0]) / v1[:-1]) + np.pi # this is to correct for BLonD
    # NOTE: the sign is flipped to better aid the optimization algorithm and to better 
    # represent the phase fo the 2nd harmonic, not of the first one.
    

    stable_phase_d = savgol_filter(stable_phase, 1000, 1) # this corresponds to 1ms sampling and linearization

    # NOTE:savgol_filter is by fer the best to remove digitization effects + noise
    # of course the parameters need to be revisited every time a new program 
    # is loaded 

    phase_diff = np.abs(np.gradient(stable_phase_d, ring.cycle_time[:-1]))

    # apply a moving average to smooth the phase_diff
    phase_diff = moving_average(phase_diff, 1000)

    # Filter the phase_diff to remove the noise
    # phase_diff = savgol_filter(phase_diff, 1000, 1)
    plt.close('all')
    a = tolSetter(phase_diff, stable_phase_d)
    tolerance = a.y
    samples = np.array(a.samples)

    peaks, _ = find_peaks(phase_diff, height = tolerance)

    eliminate = np.where(np.diff(peaks)<=1000)[0]

    if len(eliminate) == 0:
        pass
    else:
        peaks = np.delete(peaks, eliminate)

    # check if any of the peaks are within 1000 turns of any in samples
    for sample in samples:

        eliminate = np.where(np.abs(peaks - sample) <= 1000)[0]

        if len(eliminate) == 0:
            pass
        else:
            peaks = np.delete(peaks, eliminate)

    # compile the complete peaks array with the samples
    peaks = np.append(peaks, samples)
    peaks = np.sort(peaks)
    
    t_vals = ring.cycle_time[peaks]
    
    return t_vals, stable_phase

def obj_func_plotter(xs,means,stds,evals,turns):
    # Plot the the objective function at every sampled turn in a colorplot
    plt.close('all')

    fig = plt.figure()
    

    ax = fig.add_subplot(111)
    

    p = ax.scatter(xs, turns, c=means, cmap='viridis',linewidth = 0, antialiased=False)
    # obs = ax.plot(evals,turns, 'k.')
    
    ax.set_xlabel('Phase [rad]')
    ax.set_ylabel('Turn number')
    ax.set_title(r'Mean Objective Function $\mu(x)$')

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    s = ax1.scatter(xs, turns, c=stds, cmap='viridis',linewidth = 0, antialiased=False)

    ax1.set_xlabel('Phase [rad]')
    ax1.set_ylabel('Turn number')
    ax1.set_title(r'Estimate Standard Deviation Objective Function $\sigma(x)$')
    fig.colorbar(p, ax=ax)
    fig1.colorbar(s, ax=ax1)
    fig.savefig('./gpu_output_files/EX_02_fig/obj_func_mean.png')
    fig1.savefig('./gpu_output_files/EX_02_fig/obj_func_std.png')



# Define workers 
bm.use_precision('single')

# Define booleans
loading = True
unique = True
# Use interactive plots
mpl.use('TkAgg')

# Make an array defining the range of turns to simulate for that specific point
dt_opt = 20 * 1e-3 # 10 ms

# Initialize the reference beam ------------------------------------------------
# Ref beam parameters
n_particles = 2e13
n_macroparticles = 1e6
sigma_dt = 400e-9 /4  # [s]
kin_beam_energy = 160e6  # [eV] # We take the reference beam at 160 MeV

# Machine and RF parameters
radius = 25 # [m]
bend_radius = 8.239 # [m]
gamma_transition = 4.4  # [1]
C = 2 * np.pi * radius  # [m]

# Derived parameters
E_0 = m_p * c**2 / e    # [eV]
tot_beam_energy = E_0 + kin_beam_energy  # [eV]
momentum_compaction = 1 / gamma_transition**2  # [1]
sync_ref_mom = np.sqrt(tot_beam_energy**2 - E_0**2)  # [eV/c]

# RFStation Parameters
n_rf_systems = 2
# harmonic_numbers = 1
harmonic_numbers = [1,2]
voltage_program = [8e3, 5e3]  # [V]
phi_offset = np.pi

#Get the magnetic field and voltage programs for TOF and ISOHRS using pyda
# See if my optimizer still works when accessing the information from the accelerator (even when the v_2 is very low)




# Possibly in the future delete all the reference instances

# DEFINE THE ACTUAL RING + BEAM PROFILE ------------------------------------------------
# Load the momentum program
this_directory = os.path.dirname(os.path.abspath(__file__)) + '/'
sync_momentum = np.load(this_directory + '../input_files/1_4GeVOperational.npy')# in eV/c 

t_arr = sync_momentum[0] #in s
program = sync_momentum[1] * 1e-4 # in Tesla

sync_momentum = program * bend_radius *c #/1e9 # in GeV/c


total_E = np.sqrt(sync_momentum**2 + E_0**2)
kin_energy = total_E - E_0


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


# Define the ring and the optimization ranges per optimization run ----------------
ring = Ring(C, momentum_compaction, sync_momentum,
            Proton())

# Initialize the program definer
v1 = voltage_program[0] * np.ones_like(sync_momentum[0])
v2 = voltage_program[1] * np.ones_like(sync_momentum[0])
plots = ProgramDefOpt(v1,v2, time=sync_momentum[0], sync_momentum=sync_momentum[1])

v1 = plots.v1
v2 = plots.v2


voltages = [(sync_momentum[0], v1), (sync_momentum[0], v2)]

fmax=200e6
width_bin = 1/(2*fmax)
# DEFINE REFERENCE RING + BEAM PROFILE --------------------------------------------------

ref_ring = Ring(C, momentum_compaction, sync_ref_mom,
                Proton())

n_slices = np.round(ref_ring.t_rev[0] / width_bin).astype(int)

ref_beam = Beam(ref_ring, n_macroparticles, n_particles)

ref_RF_section = RFStation(ref_ring, harmonic_numbers, [v1[0],v2[0]], [np.pi, np.pi], n_rf_systems)

ref_ring_RF_section = RingAndRFTracker(ref_RF_section, ref_beam)

Full_ref_rrf = FullRingAndRF([ref_ring_RF_section])
# DEFINE REFERENCE SLICES----------------------------------------------------------------
ref_slice_beam = Profile(ref_beam, CutOptions(cut_left=0,
                                         cut_right=ref_ring.t_rev[0], n_slices=n_slices))

matched_from_distribution_function(ref_beam, Full_ref_rrf, distribution_type='parabolic_line',bunch_length=sigma_dt*4,process_pot_well = False)

ref_slice_beam.track()

(x,y) = ref_slice_beam.bin_centers , ref_slice_beam.n_macroparticles

# Filter the profile with a low pass butter-filter to remove the high frequency noise

b,a = butter(5, fmax *1/15 , 'lowpass', fs=fmax)

y = lfilter(b, a, y)


ref_slice_beam.n_macroparticles = y

interp_ref_beam = comp_prof([ref_slice_beam.bin_centers, ref_slice_beam.n_macroparticles])



# Initialize the 2nd reference RFStation for computing the array

ref_RF_section = RFStation(ring, [1], (voltages[0]), (sync_momentum[0], np.pi*np.ones_like(sync_momentum[0])), 1)
ref_voltage = ref_RF_section.voltage[0]

t_valss , phi_s1 =  range_comp(ref_voltage, ring)

turn_vals = [np.where(ring.cycle_time <= t)[0][-1] for t in t_valss]

n_slices = np.round(ring.t_rev[0] / width_bin).astype(int)

t_end = ring.cycle_time[-1]
start_turn = 0
sim_arr = []


t_arr = []
dummy = 1
dummy_2 = 1
t_count = 0 
initial = 0 
t_arr.append(0)


while t_count < total_sim_t:
    
    if initial == 0:
        t_count += dt_opt
        turn = np.where(ring.cycle_time <= t_count)[0][-1]
        sim_arr.append([dummy,turn, dummy_2])
        initial = 1
        dummy_2 = turn
    elif initial == 1:
        dummy_t = np.copy(t_count)
        t_count += dt_opt
        turn = np.where(ring.cycle_time <= t_count)[0][-1]
        cond_check = np.array([check_turn<=turn and check_turn<=dummy_2 for check_turn in turn_vals])
        if any(cond_check):

            idx_s = np.where(cond_check == True)[0]
            
            amount = len(idx_s)
            for condition_hit in range(amount):
                idx = idx_s[condition_hit]
                sim_arr.append([sim_arr[-1][2],turn, turn_vals[0]])
                dummy = turn_vals[0]
                t_arr.append(ring.cycle_time[turn_vals[0]])
                turn_vals.pop(0)
            
            sim_arr.append([sim_arr[-1][2],turn, dummy_2])
            t_arr.append(dummy_t)
            dummy = dummy_2
            initial = 2
            dummy_2 = turn
        else:
            sim_arr.append([dummy,turn, dummy_2])
            t_arr.append(dummy_t)
            dummy = dummy_2
            initial = 2
            dummy_2 = turn
    else:
        dummy_t = np.copy(t_count)
        t_count += dt_opt
        turn = np.where(ring.cycle_time <= t_count)[0][-1]
        cond_check = np.array([check_turn<=turn and check_turn<=dummy_2 for check_turn in turn_vals]) #Checks if we are in the range
        if any(cond_check):
            idx_s = np.where(cond_check == True)[0]
            
            amount = len(idx_s)
            db_counter = 0
            for condition_hit in range(amount):
                idx = idx_s[condition_hit]
                sim_arr.append([sim_arr[-1][2],turn, turn_vals[0]])
                t_arr.append(ring.cycle_time[turn_vals[0]])
                turn_vals.pop(0)
                 
            t_arr.append(dummy_t)
            sim_arr.append([sim_arr[-1][2],turn,dummy_2])
            dummy = sim_arr[-1][2]
            dummy_2 = turn
        else:
            sim_arr.append([dummy,turn, dummy_2])
            t_arr.append(dummy_t)
            dummy = dummy_2
            initial = 2
            dummy_2 = turn
    

sim_arr = np.array(sim_arr)
sim_arr[:,1] = sim_arr[:,2]
t_arr = np.array(t_arr)

# Add the turn_vals at which to force the optimizer to compute the phase

simulated_turns = 0
for entry in sim_arr:
    delta_turn = entry[1] - entry[0]
    simulated_turns+=delta_turn


# # Main loop
max_iter = 11
init_phi_est = np.pi
print('The amount of simulated turns is', simulated_turns, 'which means an aditional (simulation only) computational load of', (simulated_turns/ring.n_turns -1)*100*max_iter, '%')

global Objects

phis=[]

means = []
std = []
xs = []
evals = []

for i, entry in enumerate(sim_arr):
    
    # phi_val.value = init_phi_est
    if i == 0:
        obj_func = partial(comp_obj2,interp_ref_beam, entry, None,ring, None, None, sync_momentum , init = True,unique = True)
        phi_val = np.pi
        # phi_val = gp_minimize(obj_func, [(0, np.pi*1.05)],n_calls=max_iter, x0= [phi_val],verbose=True, acq_func="PI", n_random_starts=max_iter//5)## print(obj_func.value)
        # Here the init is set in the special run case


        print('Problem solved with value', phi_val)

        print(entry)
        Objects, phase2_arr = run_simulation(entry, None, ring, None, None, sync_momentum, phi_val, init = True, voltages=voltages)
        phis.append(phi_val)
        init_phi_est = phi_val
        
        print('{x:2f} of the beam is alive'.format(x=Objects[0].beam.n_macroparticles_alive/Objects[0].beam.n_macroparticles))

    else:
        print(entry)
        obj_func =  partial(comp_obj2, interp_ref_beam, entry, [t_arr[i], t_arr[i-1]],ring, phase2_arr[1],Objects, sync_momentum, init = False, unique = True)

        if i<=3: # Use the calculated stable phase
            phi_val = gp_minimize(obj_func, [(phi_s1[entry[2]-1] -0.4*np.pi, np.clip(phi_s1[entry[2]-1] +0.4*np.pi,a_min=None, a_max=2*np.pi))], x0= [init_phi_est], n_calls=max_iter, verbose=True, acq_func="PI" , n_random_starts=max_iter//5)## print(obj_func.value)

        else: # Use the previous value as the center-point
            phi_val = gp_minimize(obj_func, [(np.clip(phis[-1] -0.4*np.pi, a_min=0, a_max=None), np.clip(phis[-1] +0.4*np.pi, a_max=2*np.pi, a_min=None))], x0= [init_phi_est], n_calls=max_iter, verbose=True, acq_func="PI" , n_random_starts=max_iter//5)## print(obj_func.value)

                #gp_minimize(obj_func, [(0, np.pi*1.05)],n_calls=max_iter, n_random_starts=5,random_state=1234) 
        print('Problem solved with value', phi_val)

        # Get the probed values
        evals.append(np.vstack((np.array(phi_val.x_iters).flatten() , phi_val.func_vals)))

        # Get the model values for the objective function and its standard deviation
        x_i = np.linspace(phi_val.space.dimensions[0].bounds[0] , phi_val.space.dimensions[0].bounds[1],1000)
        x_model_i = phi_val.space.dimensions[0].transform(x_i)
        x_model_i = x_model_i.reshape(-1,1)
        mean_i, std_i = phi_val.models[-1].predict(x_model_i, return_std=True)

        means.append(mean_i)
        std.append(std_i)
        xs.append(x_i)

        # Run the simulation to save the objects
        Objects, phase2_arr = run_simulation(entry, [t_arr[i], t_arr[i-1]], ring, phase2_arr[1], Objects, sync_momentum, phi_val.x[0], init = False )
        phis.append(phi_val.x[0])
        init_phi_est = phi_val.x[0]
        print('{x:2f} percent of the beam is alive'.format(x=Objects[0].beam.n_macroparticles_alive/Objects[0].beam.n_macroparticles*100))

        
    


means = np.array(means)
std = np.array(std)
xs = np.array(xs)
turn_plots = np.zeros_like(means)
for i, entry in enumerate(sim_arr):
    if i ==0: 
        pass 
    else:
        turn_plots[i,:] = entry[2]
    



obj_func_plotter(xs.flatten(),means.flatten(),std.flatten(),evals,turn_plots.flatten())





# Save the phase array 
np.save('mean_obj.npy', means)
np.save('std_obj.npy', std)
np.save('x_obj.npy', xs)
np.save('phase_array.npy', np.array([phase2_arr]))
np.save('t_arr.npy' , t_arr)








