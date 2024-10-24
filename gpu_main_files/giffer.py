
import imageio
import numpy as np
import os 
# Make an animated plot of the bunch position in longitudinal space
# using the files in the path: 
this_dir = os.path.dirname(os.path.abspath(__file__))
dir_plots= this_dir + "./gpu_output_files/EX_02_fig/"

# Find all the files that start with "long_distr_" in the directory

all_images = os.listdir(dir_plots)
images_long = [x for x in all_images if x.startswith("long_distr_")]
images_beam = [x for x in all_images if x.startswith("beam_profile_")]

images_long.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
images_beam.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

# Read all the images and put them in the list
imgs_long = []
imgs_beam = []
for i in range(len(images_long)):
    imgs_long.append(imageio.imread(dir_plots + images_long[i]))




for i in range(len(images_beam)):
    imgs_beam.append(imageio.imread(dir_plots + images_beam[i]))



# Make sure all images have the same size and dimensions, if not upsample the images to the maximum size found
for i in range(len(imgs_long)):
    if i == 0:
        max_height = imgs_long[i].shape[0]
        max_width = imgs_long[i].shape[1]
    else:
        if imgs_long[i].shape[0] > max_height:
            max_height = imgs_long[i].shape[0]
        if imgs_long[i].shape[1] > max_width:
            max_width = imgs_long[i].shape[1]

for i in range(len(imgs_beam)):
    if i == 0:
        max_height_b = imgs_beam[i].shape[0]
        max_width_b = imgs_beam[i].shape[1]
    else:
        if imgs_beam[i].shape[0] > max_height_b:
            max_height_b = imgs_beam[i].shape[0]
        if imgs_beam[i].shape[1] > max_width_b:
            max_width_b = imgs_beam[i].shape[1]

for i in range(len(imgs_long)):
    if imgs_long[i].shape[0] < max_height:
        imgs_long[i] = np.pad(imgs_long[i], ((0, max_height-imgs_long[i].shape[0]), (0, 0)), 'constant')
    if imgs_long[i].shape[1] < max_width:
        imgs_long[i] = np.pad(imgs_long[i], ((0, 0), (0, max_width-imgs_long[i].shape[1])), 'constant')

imageio.mimsave(this_dir + '/gpu_output_files/EX_02_fig/long_distr.gif', imgs_long)


# Upsample all images to have the same height and width
for i in range(len(imgs_beam)):
    if imgs_beam[i].shape[0] < max_height_b:
        imgs_beam[i] = np.pad(imgs_beam[i], ((0, max_height_b-imgs_beam[i].shape[0]), (0, 0)), 'constant')
    if imgs_beam[i].shape[1] < max_width_b:
        imgs_beam[i] = np.pad(imgs_beam[i], ((0, max_width_b-imgs_beam[i].shape[1]), (0, 0)), 'constant')



imageio.mimsave(this_dir + '/gpu_output_files/EX_02_fig/beam_profile.gif', imgs_beam)



print("Done!")
