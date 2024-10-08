import numpy as np
from scipy.interpolate import interp1d
from utils_opt.run_PSB_sim_2_harmonics import *
from scipy.signal import butter,lfilter
from copy import deepcopy
# from sys import getsizeof
# import logging

# from numba import jit


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
    # Get the reference normalized profile shape in y direction
    norm_ref_profile = ref_Profile[1] / np.max(ref_Profile[1])
    
    idx = np.where(norm_ref_profile >= 0.0001)[0] # This 0.1% value might need revisiting

    # Check where there is a discontinuity in the index (because of oscillations at the end of the profile)
    # and keep the lower half of the profile
    eliminate = np.where(np.diff(idx) != 1)[0]
    
    if len(eliminate) == 0:
        pass
    else:
        idx = idx[:eliminate[0]+1] # This now has the indexes of the main lobe


    # Get the bunch length 
    bunch_array = ref_Profile[0][idx] 

    norm_ref_profile = norm_ref_profile[idx]

    bunchlength= bunch_array[-1] - bunch_array[0]

    norm_bunch_array = (bunch_array - bunch_array[0])/bunchlength

    # Interpolate the profile
    norm_ref_profile = interp1d(norm_bunch_array, norm_ref_profile)

    return norm_ref_profile

def comp_flat_index(normalized_profile):
    # Get the 3sigma  of the normalized profile
    
    integrator = np.cumsum(normalized_profile)/np.max(np.cumsum(normalized_profile))

    idxi = np.where(integrator >= 0.03)[0]

    idxf = np.where(integrator >= 0.97)[0]

    
    reference = normalized_profile[idxi[0]:idxf[0]]



    return np.sum(np.abs(reference - np.mean(reference)))




def comp_obj(normalized_ref_profile, turn_range,n_times,ring,phase_programme, Objects,sync_momentum,n_phase, final_run = False,init=False, unique = False, log_locals = False, verbose = False):
    
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

    Objects_copy = [deepcopy(Object) for Object in Objects]
    
    normalized_profiles = run_simulation_int(turn_range,n_times,ring,phase_programme, Objects_copy,sync_momentum,n_phase, unique=unique, final_run = final_run,init=init)

    
    normalized_profiless = [filter_data(normalized_profile) for normalized_profile in normalized_profiles]

    normalized_profiles_interp = [comp_prof(profile) for profile in normalized_profiless]

    for normalized_profile_i in normalized_profiles_interp:

        error += np.sum(np.abs(normalized_ref_profile(x) - normalized_profile_i(x)))
    



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