# Take in a state's pixels convert it to a numpy array and construct an image from it.

# NOTE: This is currently not implemented anywhere in the code - but was used on train and test functions
#       to get images for the assignment.

from PIL import Image
import numpy as np


# Draw an imagine from array data
def draw_image(pixels, i=0):
    np_array = np.array(pixels, dtype=np.uint8)
    new_im = Image.fromarray(np_array)
    new_im.save(f'son_im{i}.png')


# Decode normalised pixel array data and then draw
def draw_image_processed(pixels, i=0):
    np_array = np.array(pixels*255, dtype=np.uint8)
    new_im = Image.fromarray(np_array)
    new_im.save(f'im{i}_preprocessed.png')