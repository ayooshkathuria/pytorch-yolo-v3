#########################################################
# resize.py
#
# Simple script using only PIL to resize an image.
# 
# By M. Harris, 2018
#########################################################

from PIL import Image
import argparse

# Collect command line arguments
parser = argparse.ArgumentParser(description='Process command line args.')
parser.add_argument('input_img', type=str, 
                    help='Input image path')
parser.add_argument('--height', type=int,
                    help='New height in pixels')
parser.add_argument('--out', type=str,
                    help='Output image path')
args = parser.parse_args()


baseheight = args.height
img = Image.open(args.input_img)
hpercent = (baseheight / float(img.size[1]))
wsize = int((float(img.size[0]) * float(hpercent)))
img = img.resize((wsize, baseheight), Image.ANTIALIAS)
img.save(args.out)