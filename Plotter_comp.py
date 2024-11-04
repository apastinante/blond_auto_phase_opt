import numpy as np 
import matplotlib.pyplot as plt
import os
import scipy 

def obj_func_plotter(xs,means,turns, type):
    # Plot the the objective function at every sampled turn in a colorplot
    
    plt.close('all')

    fig = plt.figure()

    ax = fig.add_subplot(111)
    

    p = ax.scatter(xs, turns, c=means, cmap='viridis',linewidth = 0, antialiased=False)
    # obs = ax.plot(evals,turns, 'k.')
    
    ax.set_xlabel('Phase [rad]')
    ax.set_ylabel('Turn number')
    ax.set_title(r'Mean Objective Function $\mu(x)$ for '+ type)

    fig.colorbar(p, ax=ax)
    


home_dir = os.getcwd()
# Read the flat method 3/8 files 
mean = np.load(home_dir+'./38/mean_obj.npy')
phase_arr = np.load(home_dir+'./38/phase_array.npy')
phase_arr = np.squeeze(phase_arr)
t_arr = np.load(home_dir+'./38/t_arr.npy')
xs = np.load(home_dir+'./38/x_obj.npy')


# Read the flat method 5/8 files
mean2 = np.load(home_dir+'./58/mean_obj.npy')
phase_arr2 = np.load(home_dir+'./58/phase_array.npy')
phase_arr2 = np.squeeze(phase_arr2)
t_arr2 = np.load(home_dir+'./58/t_arr.npy')
turns2 = np.load(home_dir+'./58/turns.npy')
xs2 = np.load(home_dir+'./58/x_obj.npy')

# read the shape method unique 5/8 files 

mean3 = np.load(home_dir+'./shape_58_u/mean_obj.npy')
phase_arr3 = np.load(home_dir+'./shape_58_u/phase_array.npy')
phase_arr3 = np.squeeze(phase_arr3)
t_arr3 = np.load(home_dir+'./shape_58_u/t_arr.npy')
turns3 = np.load(home_dir+'./shape_58_u/turns.npy')
xs3 = np.load(home_dir+'./shape_58_u/x_obj.npy')

# read the shape method multi 5/8 files

mean4 = np.load(home_dir+'./shape_58_full/mean_obj.npy')
phase_arr4 = np.load(home_dir+'./shape_58_full/phase_array.npy')
phase_arr4 = np.squeeze(phase_arr4)
t_arr4 = np.load(home_dir+'./shape_58_full/t_arr.npy')
turns4 = np.load(home_dir+'./shape_58_full/turns.npy')
xs4 = np.load(home_dir+'./shape_58_full/x_obj.npy')


# read the MD3 files

mean5 = np.load(home_dir+'./MD3/mean_obj.npy')
phase_arr5 = np.load(home_dir+'./MD3/phase_array.npy')
phase_arr5 = np.squeeze(phase_arr5)
t_arr5 = np.load(home_dir+'./MD3/t_arr.npy')
turns5 = np.load(home_dir+'./MD3/turns.npy')
xs5 = np.load(home_dir+'./MD3/x_obj.npy')
percent_alives5 = np.load(home_dir+'./MD3/percent_alives.npy')

# read the MD3_improved files

mean6 = np.load(home_dir+'./MD3_improved/mean_obj.npy')
phase_arr6 = np.load(home_dir+'./MD3_improved/phase_array.npy')
phase_arr6 = np.squeeze(phase_arr6)
t_arr6 = np.load(home_dir+'./MD3_improved/t_arr.npy')
turns6 = np.load(home_dir+'./MD3_improved/turns.npy')    
xs6 = np.load(home_dir+'./MD3_improved/x_obj.npy')
percent_alives6 = np.load(home_dir+'./MD3_improved/percent_alives.npy')

fig404, ax404 = plt.subplots(1,1)

ax404.plot(phase_arr5[0],phase_arr5[1], label = 'MD3')
ax404.plot(phase_arr6[0],phase_arr6[1], label = 'MD3_improved')
ax404.set_xlabel('Time [ms]')
ax404.set_ylabel('Phase [rad]')
ax404.set_title('Optimized 2nd Harmonic Absolute Phase $\Phi_2$ for MD3 and MD3_improved')
ax404.legend()

fig405, ax405 = plt.subplots(1,1)

ax405.plot(phase_arr5[0],percent_alives5, label = 'MD3')
ax405.plot(phase_arr6[0],percent_alives6, label = 'MD3 improved')
ax405.set_xlabel('Time [ms]')
ax405.set_ylabel('Percentage of Particles in Bucket')
ax405.set_title('Percentage of Particles in Bucket for MD3 and MD3_improved')
ax405.legend()

fig , ax = plt.subplots(1,1)

ax.plot(phase_arr[0],phase_arr[1], label = r'$r=\frac{3}{8}$')
ax.plot(phase_arr2[0],phase_arr2[1], label = r'$r=\frac{5}{8}$')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Phase [rad]')
ax.set_title(r'Optimized 2nd Harmonic Absolute Phase $\Phi_2$ for Method 2 with Multiple Profiles')
ax.legend()

fig2 , ax2 = plt.subplots(1,1)

ax2.plot(phase_arr3[0],phase_arr3[1], label = r'Method 1: Single Profile')
ax2.plot(phase_arr4[0],phase_arr4[1], label = r'Method 1: Multiple Profiles')
ax2.plot(phase_arr2[0],phase_arr2[1], label = r'Method 2: Multiple Profiles$')
ax2.set_xlabel('Time [ms]')
ax2.set_ylabel('Phase [rad]')
ax2.set_title(r'Optimized 2nd Harmonic Absolute Phase $\Phi_2$ for $r=\frac{5}{8}$')
ax2.legend()



# For each of the mean objective timeseries, normalize the mean function by the max of the mean 
# at that specific turn

means = [ mean, mean2, mean3, mean4]
xs_s = [xs, xs2, xs3, xs4]
turns_s = [turns2, turns2, turns3, turns4]
types = [r'Method 2: $r=\frac{3}{8}$', r'Method 2: $r=\frac{5}{8}$', r'Method 1: Single Profile', r'Method 1: Multiple Profiles']
for j,m in enumerate(means):
    for i in range(len(m)):
        m[i, :] = m[i,:]/np.max(m[i,:])

    obj_func_plotter(xs_s[j],m,turns_s[j], types[j])

plt.show()



