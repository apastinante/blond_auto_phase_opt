# Disclaimer 
This is an automatic phase optimizer developed for a double harmonic system and specifically using the modelled parameters of the Proton Synchrotron Booster at CERN. The project was completed at CERN for a duration of 3 months as part of the course AE5051 Internship for the Double MSc in Space Engineering and Physics for Instrumentation.


# Requirements
In order to run this code, one will need a GPU, unless the USE_GPU Boolean in both functions within "\gpu_main_files\utils_opt\run_PSB_sim_2_harmonics.py" is set to False. 

The following packages will need to be installed: 
- cuda 
- cupy
- numpy 
- blond
- matplotlib
- skopt 
- scipy
- functools 

# Instructions

The code has to be run from the "\gpu_main_files\main.py" file and one must at least set one additional sample and a tolerance level in the Stable Phase and Delta E interactive plotters, respectively. 

The characteristics of the accelerator can also be modified in principle and the optimizer should still converge. 
