#########################################################
# padding.py
#
# Simple script using only numpy and matplotlib to add 
# padding and create a square image.
# 
# By M. Harris, 2018
#########################################################

import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# Collect command line arguments
parser = argparse.ArgumentParser(description='Process command line args.')
parser.add_argument('--input_dir', type=str,
                    help='Images directory')
parser.add_argument('--output_dir', type=str,
                    help='Padded images output directory')

args = parser.parse_args()

def pad_image(np_img, new_img_file):
    """Use numpy operations to add padding around an image in
    numpy format (an array of rank 3 - so 3 channels) in order
    to create a square image.  Function saves the newly padded
    image to a new file in the output directory specified.
    
    Args
    ----
      np_img : numpy array
          image
      new_img_file : str
          new file location
    """

    h, w, c = np_img.shape
    side_len = max(h, w)
    # Create our square "palette" or area upon which the image data is placed
    # Make it kinda grey (e.g. a palette of all > 1)
    new_np_img = np.ones(side_len * side_len * c).reshape(side_len, 
        side_len, c) * 100

    if h > w:
        for i in range(c):
            # Multiply by 255 because plt read in as vals 0-1, not 0-255
            old_patch = np_img[:, :, i] * 255
            pad = (side_len - w)//2
            new_np_img[:, pad:(pad + w), i] = old_patch
        plt.imsave(new_img_file, new_np_img)
    elif w > h:
        for i in range(c):
            old_patch = np_img[:, :, i] * 255
            pad = (side_len - h)//2
            new_np_img[pad:(pad + h), :, i] = old_patch
        plt.imsave(new_img_file, new_np_img)
    else:
        # Image already square - lucky!
        pass


def main():
    """Loops through all image in given input directory.  Reads
    each file in a try/except block in the instance the file
    is not an image (it is then skipped).

    Calls a function to add padding to square up the image
    and saves it to a new file.
    """

    # Just grab all files - we'll use try/except to filter
    images = glob.glob(os.path.join(args.input_dir, '*.*'))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for img_file in images:
        print(img_file)
        try:
            np_img = plt.imread(img_file)
            print(np_img.shape)
            img_name = img_file.split(os.sep)[-1]
            new_img_file = os.path.join(args.output_dir, img_name)
            pad_image(np_img, new_img_file)
        except Exception as e:
            print('Warning:  {}.  Skpping file.'.format(e))
            continue

if __name__ == '__main__':
    main()
    



