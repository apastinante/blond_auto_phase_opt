
import imageio
import numpy as np

# Make an animated plot of the bunch position in longitudinal space
# using the files in the path: 
dir_plots= "/home/apastinante/BLonD-master-__EXAMPLES/__EXAMPLES/gpu_output_files/EX_02_fig/"

# Make the animation and display it as an html using the animation class

# Read all the images and put them in the list
imgs = []
k=0
try:
    while True:
        

        dir = dir_plots + "long_distr_" + str(k) + ".png"
        imgs.append(imageio.v3.imread(dir))
        k+=1

except:
    pass 



imageio.mimsave('/home/apastinante/BLonD-master-__EXAMPLES/__EXAMPLES/gpu_output_files/EX_02_fig/long_distr.gif', imgs)




print("Done!")
