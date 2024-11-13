import numpy as np
from scipy.interpolate import interp1d
if __name__ == '__main__':
    from run_PSB_sim_2_harmonics import *
else:
    from .run_PSB_sim_2_harmonics import *
from scipy.signal import butter,lfilter
from copy import deepcopy
import time 
# from sys import getsizeof
# import logging

# from numba import jit


def comp_phase_array_multi(time_arr,phase_array,phase_n,n_times):
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

def filter_data(Profile):
    # Apply a filter to remove the noise in the data: 
    fmax = 200e6

    b,a = butter(5, fmax *1/15 , 'lowpass', fs=fmax)

    y = lfilter(b, a, Profile[1])

    return Profile[0], y
# @jit
def comp_prof_gpu(ref_Profile):

    # Get the reference normalized profile shape in y direction
    norm_ref_profile = ref_Profile.n_macroparticles.get() / np.max(ref_Profile.n_macroparticles.get())
    
    idx = np.where(norm_ref_profile >= 0.001)[0] # This 0.1% value might need revisiting

    # Check where there is a discontinuity in the index (because of oscillations at the end of the profile)
    # and keep the lower half of the profile
    eliminate = np.where(np.diff(idx) != 1)[0][0]

    idx = idx[:eliminate+1] # This now has the indexes of the main lobe

    # Get the bunch length 
    bunch_array = ref_Profile.bin_centers.get()[idx] 

    norm_ref_profile = norm_ref_profile[idx]

    bunchlength= bunch_array[-1] - bunch_array[0]

    norm_bunch_array = (bunch_array - bunch_array[0])/bunchlength

    # Interpolate the profile
    norm_ref_profile = interp1d(norm_bunch_array, norm_ref_profile, ref_Profile.bin_centers.get())

    return norm_ref_profile

def comp_prof(ref_Profile):
    """NOTE
    Another idea is to divide by the total amount of particles, 
    and then take the cumulative 5% off of either side, such that 
    we only keep 95% or any other amount (could be 2 or 3 sigma to 
    represent the entire bunch)
    Currently it trims anything below 0.1% of the max
    """
# CURRENT METHOD 
    # Get the reference normalized profile shape in y direction
    norm_ref_profile = np.abs(ref_Profile[1]) / np.max(ref_Profile[1])
    
    # idx = np.where(norm_ref_profile >= 0.001)[0] # THIS IS WHAT CAUSES ISSUES OF LOCATION OF THE PHASE VALUE! 

    integrator = np.cumsum(norm_ref_profile)/np.sum(norm_ref_profile)
    # integrator = np.cumsum(norm_ref_profile)/np.max(np.cumsum(norm_ref_profile))
    idxi = np.where(integrator >= 0.0001)[0]

    idxf = np.where(integrator >= 0.9999)[0]

    
    norm_ref_profile = norm_ref_profile[idxi[0]:idxf[0]]


    # Check where there is a discontinuity in the index (because of oscillations at the end of the profile)
    # and keep the lower half of the profile
    # eliminate = np.where(np.diff(idx) != 1)[0]
    
    # if len(eliminate) == 0:
    #     pass
    # else:
    #     idx = idx[:eliminate[0]+1] # This now has the indexes of the main lobe

    # # Get the bunch length 
    bunch_array = ref_Profile[0][idxi[0]:idxf[0]] 

# #PREVIOUS METHOD 
#    # Get the reference normalized profile shape in y direction
#     norm_ref_profile = np.abs(ref_Profile[1]) / np.max(ref_Profile[1])
    
#     idx = np.where(norm_ref_profile >= 0.0001)[0] # This 0.1% value might need revisiting

#     # Check where there is a discontinuity in the index (because of oscillations at the end of the profile)
#     # and keep the lower half of the profile
#     eliminate = np.where(np.diff(idx) != 1)[0]
    
#     if len(eliminate) == 0:
#         pass
#     else:
#         idx = idx[:eliminate[0]+1] # This now has the indexes of the main lobe


#     # Get the bunch length 
#     bunch_array = ref_Profile[0][idx] 

#     norm_ref_profile = norm_ref_profile[idx]

    bunchlength= bunch_array[-1] - bunch_array[0]

    norm_bunch_array = (bunch_array - bunch_array[0])/bunchlength

    # Interpolate the profile
    norm_ref_profile = interp1d(norm_bunch_array, norm_ref_profile)

    return norm_ref_profile

def comp_prof_multi(ref_Profile):
    """NOTE
    Another idea is to divide by the total amount of particles, 
    and then take the cumulative 5% off of either side, such that 
    we only keep 95% or any other amount (could be 2 or 3 sigma to 
    represent the entire bunch)
    Currently it trims anything below 0.1% of the max
    """
    # Get the reference normalized profile shape in y direction
    norm_ref_profile = np.abs(ref_Profile[1]) / np.max(ref_Profile[1])
    
    # idx = np.where(norm_ref_profile >= 0.001)[0] # THIS IS WHAT CAUSES ISSUES OF LOCATION OF THE PHASE VALUE! 

    integrator = np.cumsum(norm_ref_profile)/np.sum(norm_ref_profile)
    # integrator = np.cumsum(norm_ref_profile)/np.max(np.cumsum(norm_ref_profile))
    idxi = np.where(integrator >= 0.01)[0]

    idxf = np.where(integrator >= 0.99)[0]

    
    norm_ref_profile = norm_ref_profile[idxi[0]:idxf[0]]


    # # Get the bunch length 
    bunch_array = ref_Profile[0][idxi[0]:idxf[0]] 


    bunchlength= bunch_array[-1] - bunch_array[0]

    norm_bunch_array = (bunch_array - bunch_array[0])/bunchlength

    # Interpolate the profile
    norm_ref_profile = interp1d(norm_bunch_array, norm_ref_profile)

    return norm_ref_profile

def comp_flat_index(normalized_profile):
    # Get the 3sigma  of the normalized profile
    
    integrator = np.cumsum(normalized_profile)/np.max(np.sum(normalized_profile))

    idxi = np.where(integrator >= 0.00001)[0]

    idxf = np.where(integrator >= 0.99999)[0]

    
    reference = normalized_profile[idxi[0]:idxf[0]]



    return np.sum(np.abs(reference - np.mean(reference)))




def comp_obj(normalized_ref_profile, turn_range,n_times,ring,phase_programme, Objects,sync_momentum,n_phase, final_run = False,init=False, unique = False, log_locals = False, verbose = False, voltages = None):
    
    """
    Compute the difference between the normalized reference profile and the profiles
    computed through the tracking simulation at the given turn numbers.

    Parameters
    ----------
    normalized_ref_profile: callable
        A callable that takes a numpy array of x values between 0 and 1 and returns
        the normalized reference profile at those x values.

    turn_range: list of 3 integers
        The range over which to simulate, the first two integers are the start and end
        of the range, the third integer is the number of turns at which to compute the
        phase.

    n_times: list of 2 floats
        The time at which the phase is computed (n_times[1]) and the time of the
        previous phase computation point (n_times[0])

    ring: Ring
        The ring to simulate

    phase_programme: tuple of (time, phase) arrays
        The phase programme to use

    Objects: list of tracking objects
        The list of tracking objects to use

    sync_momentum: tuple of arrays
        The sync momentum to use (time, momentum)

    n_phase: float
        The phase to use at n_times[1] in the phase programme

    final_run: bool
        If the function is called at the end of the optimization

    init: bool
        If the function is called at the beginning of the optimization

    unique: bool
        If only one trace is generated at n_time, if False then the optimizer will
        generate multiple traces and the average error over the entire turn range will
        be returned

    log_locals: bool
        If True, log the size of the objects

    verbose: bool
        If True, print some debug information

    Returns
    -------
    error: float
        The difference between the normalized reference profile and the profiles
        computed through the tracking simulation at the given turn numbers
    """
    x = np.linspace(0,1,1000)
    error = 0
    # # Log the size of the objects
    # loc = locals()
    # for element in locals():
    #     logging.debug(f'INTERNALS: {element} : {getsizeof(loc[element])}')

    # logging.debug('RFR:{x}, Profile: {y}, IndV: {z}, {turn}'.format(x = getsizeof(Objects[0]),y = getsizeof(Objects[1]),z = getsizeof(Objects[2]) , turn = turn_range[1]))
    if init:
        Objects_copy = None
    else:
        Objects_copy = [deepcopy(Object) for Object in Objects]
    
    
    [normalized_profiles , corrections] = run_simulation_int(turn_range,n_times,ring,phase_programme, Objects_copy,sync_momentum,n_phase, unique=unique, final_run = final_run,init=init, voltages=voltages)
    
    
    normalized_profiless = [filter_data(normalized_profile) for normalized_profile in normalized_profiles]

    normalized_profiles_interp = [comp_prof(profile) for profile in normalized_profiless]

    for ind ,normalized_profile_i in enumerate(normalized_profiles_interp):

        error += np.sum(np.abs(normalized_ref_profile(x) - normalized_profile_i(x)))/corrections[ind]
    



    return error/len(normalized_profiles)

def comp_obj_multi(normalized_ref_profile, turn_ranges,n_times_list,ring, Objects,sync_momentum,n_phases, final_run = False,init=False, unique = False, log_locals = False, verbose = False, voltages = None):
    
    """
    Compute the difference between the normalized reference profile and the profiles
    computed through the tracking simulation at the given turn numbers.

    Parameters
    ----------
    normalized_ref_profile: callable
        A callable that takes a numpy array of x values between 0 and 1 and returns
        the normalized reference profile at those x values.

    turn_ranges: list of lists of 3 integers
        The ranges over which to simulate, the first two integers are the start and end
        of the range, the third integer is the number of turns at which to compute the
        phase.

    n_times: list of lists of 2 floats
        The time at which the phase is computed (n_times[1]) and the time of the
        previous phase computation point (n_times[0])

    ring: Ring
        The ring to simulate

    phase_programme: tuple of (time, phase) arrays
        The phase programme to use

    Objects: list of tracking objects
        The list of tracking objects to use

    sync_momentum: tuple of arrays
        The sync momentum to use (time, momentum)

    n_phase: float
        The phase to use at n_times[1] in the phase programme

    final_run: bool
        If the function is called at the end of the optimization

    init: bool
        If the function is called at the beginning of the optimization

    unique: bool
        If only one trace is generated at n_time, if False then the optimizer will
        generate multiple traces and the average error over the entire turn range will
        be returned

    log_locals: bool
        If True, log the size of the objects

    verbose: bool
        If True, print some debug information

    Returns
    -------
    error: float
        The difference between the normalized reference profile and the profiles
        computed through the tracking simulation at the given turn numbers
    """
    x = np.linspace(0,1,1000)
    error = 0
    if init:
        Objects_copy = None
    else:
        Objects_copy = [deepcopy(Object) for Object in Objects]
    turn_range_full = [turn_ranges[0][0], turn_ranges[-1][1], turn_ranges[-1][1]]
    n_times = n_times_list[0]
    
    for i, turn_range in enumerate(turn_ranges):
        if i == 0:
            phase_programme_iter = n_phases[i]*np.ones_like(sync_momentum[0])
        else:
            phase_programme_iter = comp_phase_array_multi(sync_momentum[0], phase_programme_iter,n_phases[i], [n_times_list[i], n_times_list[i-1]])

    result = run_simulation_int(turn_range_full,n_times,ring,(sync_momentum[0],phase_programme_iter), Objects_copy,sync_momentum,n_phases[0], unique=unique, final_run = final_run,init=init, voltages=voltages, full_phase=True)
    
    normalized_profiles , corrections, flag = result
    
    normalized_profiless = [filter_data(normalized_profile) for normalized_profile in normalized_profiles]

    normalized_profiles_interp = [comp_prof_multi(profile) for profile in normalized_profiless]

    for ind, normalized_profile_i in enumerate(normalized_profiles_interp):
        error += np.sum(np.abs(normalized_ref_profile(x) - normalized_profile_i(x)))/corrections[ind]
    
    if flag==0:
        error *= 1e3
    
    return error/len(normalized_profiles)


def comp_obj2(normalized_ref_profile, turn_range,n_times,ring,phase_programme, Objects,sync_momentum,n_phase, final_run = False,init=False, unique = False, log_locals = False, verbose = False):
    
    x = np.linspace(0,1,1000)
    error = 0
    # # Log the size of the objects
    # loc = locals()
    # for element in locals():
    #     logging.debug(f'INTERNALS: {element} : {getsizeof(loc[element])}')

    # logging.debug('RFR:{x}, Profile: {y}, IndV: {z}, {turn}'.format(x = getsizeof(Objects[0]),y = getsizeof(Objects[1]),z = getsizeof(Objects[2]) , turn = turn_range[1]))

    Objects_copy = [deepcopy(Object) for Object in Objects]
    
   
    normalized_profiles = run_simulation_int(turn_range,n_times,ring,phase_programme, Objects_copy,sync_momentum,n_phase, unique=unique, final_run = final_run,init=init)
    

    # normalized_profiless = [filter_data(normalized_profile) for normalized_profile in normalized_profiles]

    # normalized_profiles_interp = [comp_prof(profile) for profile in normalized_profiless]
    goal = comp_flat_index(normalized_ref_profile(x)) # Should retrieve the best value

    for normalized_profile_i in normalized_profiles:

        error += np.abs(goal - comp_flat_index(normalized_profile_i[1]))

    return error/len(normalized_profiles)




# @jit
# @jit mk,n.lj;/'hbihbgnjmkkmjn.,hlbg/;uivyft'd rmnknmk,jb.hl/ ;gvuifnmhgbjnvkhgbjnhjngkbmvflu,d.nbmhjk,gv.l hbnbmhjk,hnjbkmnmbjk,h.lnmbjk,h.mkn,j.lbh;/g iuv'bh/knljb;h '


# """
# Objects that will be only loaded once: 
# Ring ()
 
# Object that need to be loaded at the beginning of the optimization run:
# RFStation
# Beam (does this also need to be matched? I think not) 
# RingAndRFTracker
# FullRingAndRF
# Profile 
# (BunchMonitor)
# (InducedVoltageFreq/TotalInducedVoltage)?


# """
# The idea is to take a certain
#  timestep over which CVXPY will optimize the phase for that timestep
# The timestep can be changed to vary performance, then a function can interpolate a function over 
# all points and get the optimum phase to yield the desired beam profile at that timestep

# The objective function has to be the comparison of the shapes of the reference and the profile at the given timestep

# At the start, let it advance 100 turns (or the timestep), then chose the point such that a linear phase ramp is 
# used between the timestep between the previous phase and the current phase (being optimized)
if __name__ == "__main__":
    import blond.utils.bmath as bm
    from blond.beam.beam import Beam, Proton
    from blond.beam.distributions import matched_from_distribution_function
    from blond.beam.profile import CutOptions, Profile
    from blond.input_parameters.rf_parameters import RFStation
    from blond.input_parameters.ring import Ring
    from blond.input_parameters.ring import Ring
    from blond.trackers.tracker import FullRingAndRF,RingAndRFTracker

    # Define workers 
    bm.use_precision('single')
    bm.use_py()

    # Define booleans
    loading = True
    unique = True
    # Use interactive plots
    mpl.use('TkAgg')

    # Make an array defining the range of turns to simulate for that specific point
    dt_opt = 20 * 1e-3 # 10 ms

    # Initialize the reference beam ------------------------------------------------
    # Ref beam parameters
    n_particles = 0.9e13
    n_macroparticles = 1e6
    sigma_dt = 700e-9 /4  # [s]
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
    voltage_program = [8e3, 3e3]  # [V]
    phi_offset = np.pi

    #Get the magnetic field and voltage programs for TOF and ISOHRS using pyda
    # See if my optimizer still works when accessing the information from the accelerator (even when the v_2 is very low)




    # Possibly in the future delete all the reference instances

    # DEFINE THE ACTUAL RING + BEAM PROFILE ------------------------------------------------
    # Load the momentum program
    this_directory = os.path.dirname(os.path.abspath(__file__)) + '/'
    sync_momentum2 = np.load(this_directory + '../../input_files/1_4GeVOperational.npy')# in eV/c 
    sync_momentum = np.load(this_directory + '../../v_programs_dt/MD3/B2GeV.npy')# in eV/c 
    B = deepcopy(sync_momentum[1])
    
   


    
    t_arr = sync_momentum[0] #in s
    program = sync_momentum[1] * 1e-4 # in Tesla

    sync_momentum = program * bend_radius *c #/1e9 # in GeV/c


    total_E = np.sqrt(sync_momentum**2 + E_0**2)
    kin_energy = total_E - E_0


    inj_t = 275  # in s
    ext_t = 805  # in s

    # Compute the derivarive of B to get B_dot
    
    # Find the maximum index of B_dot 


    index_inj = np.where(t_arr <= inj_t)[0][-1]
    index_ext = np.where(t_arr >= ext_t)[0][0]

    t_arr = t_arr[index_inj:index_ext]
    t_arr = (t_arr-t_arr[0])/1e3
    sync_momentum = sync_momentum[index_inj:index_ext]
    kin_energy = kin_energy[index_inj:index_ext]

    B = B[index_inj:index_ext]
    B_dot = np.gradient(B, t_arr)
    ind_max = np.argmax(B_dot)

    plt.figure(1)
    plt.plot(t_arr, B)
    plt.scatter(t_arr[ind_max], B[ind_max], c='r')
    plt.show(block=True)
    plt.close()

    sync_momentum = sync_momentum[ind_max:]
    kin_energy = kin_energy[ind_max:]
    t_arr = t_arr[ind_max:]

    t_range = [inj_t, ext_t]
    total_sim_t = (ext_t - inj_t)/1000
    t_range = [0,t_arr[-1]]
    dt = 0.1 # in ms for the timeseries

    sync_momentum = (t_arr, sync_momentum)
    kin_energy = (t_arr, kin_energy)

    # Define the ring and the optimization ranges per optimization run ----------------
    ring = Ring(C, momentum_compaction, sync_momentum,
                Proton())

    fmax=200e6
    width_bin = 1/(2*fmax)
    # DEFINE REFERENCE RING + BEAM PROFILE --------------------------------------------------

    ref_ring = Ring(C, momentum_compaction, sync_momentum,
                    Proton())
    ref_coasting_ring = Ring(C, momentum_compaction, sync_ref_mom,
                             Proton())
    
    n_slices = np.round(ref_ring.t_rev[0] / width_bin).astype(int)
    n_slices_coasting = np.round(ref_coasting_ring.t_rev[0] / width_bin).astype(int)
    ref_beam = Beam(ref_ring, n_macroparticles, n_particles)
    ref_beam_coasting = Beam(ref_coasting_ring, n_macroparticles, n_particles)

    




    # Produce a grid of different total voltages and voltage ratios
    ph2 = np.array([np.hstack((np.pi/2,np.linspace(0.65*np.pi, 0.75*np.pi, 10))), np.hstack((np.pi/2,np.linspace(0.6*np.pi, 0.7*np.pi, 10))), np.hstack((np.pi/2,np.linspace(0.51*np.pi, 0.61*np.pi, 10)))])
    
    ph1 = np.pi
    x_arr = np.linspace(0, 1, 1000)
    params = []
    total_voltages = [14e3]
    v_ratios = [3/8,4/8,5/8]

    # Approximate bucket area
    phi_s_app = ref_ring.delta_E[0][0]/total_voltages[0]
    A_bk = 8/(2*np.pi*1/ref_ring.t_rev[0]) * (1-phi_s_app)/(1+phi_s_app) * np.sqrt((2*total_voltages[0]*ref_ring.beta[0][0]**2 * ref_ring.energy[0][0])/(2*np.pi*np.abs(1/gamma_transition**2 - 1/ref_ring.gamma[0][0]**2)))
    print(A_bk)


    emittances = np.linspace(1,A_bk,7, endpoint=True)

    max_obj_val = np.zeros((len(emittances), len(v_ratios)))
    interp_beams = []
    
    for main_index,total_voltage in enumerate(total_voltages):
        interp_beams_i=[]
        for indicey, v_ratio in enumerate(v_ratios):
                v1 = total_voltage/(1+v_ratio)
                v2 = v1*v_ratio
                ref_RF_section = RFStation(ref_coasting_ring, harmonic_numbers, [v1,v2], [np.pi, np.pi], n_rf_systems)

                ref_ring_RF_section = RingAndRFTracker(ref_RF_section, ref_beam_coasting)

                Full_ref_rrf = FullRingAndRF([ref_ring_RF_section])
                # DEFINE REFERENCE SLICES----------------------------------------------------------------
                ref_slice_beam = Profile(ref_beam_coasting, CutOptions(cut_left=0,
                                                        cut_right=ref_coasting_ring.t_rev[0], n_slices=n_slices_coasting))

                matched_from_distribution_function(ref_beam_coasting, Full_ref_rrf, distribution_type='parabolic_line',bunch_length=sigma_dt*4,process_pot_well = False)

                ref_slice_beam.track()

                x,y = ref_slice_beam.bin_centers , ref_slice_beam.n_macroparticles
                
                b,a = butter(1, fmax *1/40 , 'lowpass', fs=fmax)

                y = lfilter(b, a, y)

                interp_ref_beam = comp_prof([ref_slice_beam.bin_centers, y])
                interp_beams_i.append(interp_ref_beam)

                for indicex, phi in enumerate(ph2[indicey,:]):
                    
                    
                    RF_section = RFStation(ref_ring, harmonic_numbers, [v1,v2], [np.pi, phi], n_rf_systems)

                    ring_RF_section = RingAndRFTracker(RF_section, ref_beam)

                    Full_rrf = FullRingAndRF([ring_RF_section])
                    # DEFINE REFERENCE SLICES----------------------------------------------------------------
                    
                    for indicez, emittance in enumerate(emittances):
                        print(ref_ring.t_rev[0])
                        
                        slice_beam = Profile(ref_beam, CutOptions(cut_left=0,
                                                            cut_right=ref_ring.t_rev[0], n_slices=n_slices))
                        
                        matched_from_distribution_function(ref_beam, Full_rrf, distribution_type='parabolic_line',emittance = emittance,process_pot_well = False)

                        slice_beam.track()
                        
                        x,y = slice_beam.bin_centers , slice_beam.n_macroparticles

                        norm_prof = filter_data([x,y]) 

                        norm_interp = comp_prof(norm_prof)

                        # Normalize x and y
                        x = (x - x[0])/(x[-1] - x[0])
                        y = y/(np.max(y))

                        obj_val = np.sum(np.abs(interp_ref_beam(x_arr) - norm_interp(x_arr)))

                        if obj_val > max_obj_val[indicez,indicey]:
                            max_obj_val[indicez,indicey] = obj_val
                        
                        params.append(dict(totalvoltage = total_voltage, v_ratio = v_ratio, profile = (x,y), phi2=phi , emittance = emittance, obj_val = obj_val, indices = (indicex,indicey,indicez)))
        interp_beams.append(interp_beams_i)
    # Make a grid of subplots, there should be len(ph2) in the x axis of the sublots
    # in the y axis there should be the 3 different voltage ratios 

    

    for fig_ind , emittance in enumerate(emittances):
        fig, ax = plt.subplots(3, len(ph2[0])+1, sharex=True, sharey=True)

        # Make each column be titled by the ph2 value 
        for z in range(len(v_ratios)):
            for i, phi in enumerate(ph2[z,:]):
                ax[z,i].set_title(r'$\Phi_2$ = {:.2f}'.format(phi))
            ax[z,0].set_ylabel('V1/V2 = {:.2f}'.format(v_ratios[z]))

        for ind, entry in enumerate(params):
            if entry['emittance'] == emittance:
                ax[entry['indices'][1],entry['indices'][0]].plot(entry['profile'][0], entry['profile'][1], color=plt.cm.viridis(entry['obj_val']/max_obj_val[entry['indices'][2],entry['indices'][1]]))
        
        ax[0,-1].set_title('Reference')
        ax[0,-1].plot(x_arr, interp_beams[0][0](x_arr), 'k')
        ax[1,-1].plot(x_arr, interp_beams[0][1](x_arr), 'k')
        ax[2,-1].plot(x_arr, interp_beams[0][2](x_arr), 'k')

        fig.suptitle('Profile + Obj_func for emittance = {:.2f}'.format(emittance))

plt.show()


    # fig1, ax1 = plt.subplots(3, len(ph2)+1, sharex=True, sharey=True)
    # # Make each column be titled by the ph2 value 
    # for i in range(len(ph2)):
    #     ax1[0,i].set_title(r'$\Phi_2$ = {:.2f}'.format(ph2[i]))
    # for i in range(len(v_ratios)):
    #     ax1[i,0].set_ylabel('V1/V2 = {:.2f}'.format(v_ratios[i]))

    # for ind, entry in enumerate(params):
    #     if entry['emittance'] == emittances[0]:
    #         ax1[entry['indices'][1],entry['indices'][0]].plot(entry['profile'][0], entry['profile'][1], color=plt.cm.viridis(entry['obj_val']/max_obj_val[0,entry['indices'][1]]))
    
    # ax1[0,-1].set_title('Reference')
    # ax1[0,-1].plot(x_arr, interp_beams[0][0](x_arr), 'k')
    # ax1[1,-1].plot(x_arr, interp_beams[0][1](x_arr), 'k')
    # ax1[2,-1].plot(x_arr, interp_beams[0][2](x_arr), 'k')

    # fig1.suptitle('Emmitance = {} eVs'.format(emittances[0]))

    # fig2, ax2 = plt.subplots(3, len(ph2)+1, sharex=True, sharey=True)
    # for i in range(len(ph2)):
    #     ax2[0,i].set_title(r'$\Phi_2$ = {:.2f}'.format(ph2[i]))
    # for i in range(len(v_ratios)):
    #     ax2[i,0].set_ylabel('V1/V2 = {:.2f}'.format(v_ratios[i]))

    # for ind, entry in enumerate(params):
    #     if entry['emittance'] == emittances[1]:
    #         ax2[entry['indices'][1],entry['indices'][0]].plot(entry['profile'][0], entry['profile'][1], color=plt.cm.viridis(entry['obj_val']/max_obj_val[1,entry['indices'][1]]))

    # ax2[0,-1].set_title('Reference')
    # ax2[0,-1].plot(x_arr, interp_beams[0][0](x_arr), 'k')
    # ax2[1,-1].plot(x_arr, interp_beams[0][1](x_arr), 'k')
    # ax2[2,-1].plot(x_arr, interp_beams[0][2](x_arr), 'k')

    # fig2.suptitle('Emmitance = {} eVs'.format(emittances[1]))

    # fig3, ax3 = plt.subplots(3, len(ph2)+1, sharex=True, sharey=True)
    # for i in range(len(ph2)):
    #     ax3[0,i].set_title(r'$\Phi_2$ = {:.2f}'.format(ph2[i]))
    # for i in range(len(v_ratios)):
    #     ax3[i,0].set_ylabel('V1/V2 = {:.2f}'.format(v_ratios[i]))

    # for ind, entry in enumerate(params):
    #     if entry['emittance'] == emittances[2]:
    #         ax3[entry['indices'][1],entry['indices'][0]].plot(entry['profile'][0], entry['profile'][1], color=plt.cm.viridis(entry['obj_val']/max_obj_val[2, entry['indices'][1]]))

    # ax3[0,-1].set_title('Reference')
    # ax3[0,-1].plot(x_arr, interp_beams[0][0](x_arr), 'k')
    # ax3[1,-1].plot(x_arr, interp_beams[0][1](x_arr), 'k')
    # ax3[2,-1].plot(x_arr, interp_beams[0][2](x_arr), 'k')

    # fig3.suptitle('Emmitance = {} eVs'.format(emittances[2]))

    # plt.show()




            
