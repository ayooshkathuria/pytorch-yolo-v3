"""
Convert labels from the VoTT YOLO format to VoTT Tensorflow Pascal VOC format
so that we can run kmeans.py to discover anchor sizes.

This script is only used here in conjunction with kmeans.py and is not
utilized in training or evaluating the model.

Note the use of hardcoded image name suffixes in "gather_bboxes"!

e.g.
From (<object-class> <x_center> <y_center> <width> <height>)

1 0.244375 0.473346 0.028750 0.049632
1 0.438750 0.373162 0.025000 0.036765

To (<xmin>,<ymin>,<xmax>,<ymax>,<object-class>)

path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
path/to/img2.jpg 120,300,250,600,2
"""

import argparse
import glob
from PIL import Image
import os


def get_img_dims(img):
    """returns (width, height)"""
    img = Image.open(img)
    return img.size

def center_to_x1y1x2y2(bbox, width, height):
    """Convert:
    (x_center, y_center, width, height)
    to
    (x_min, y_min, x_max, y_max)
    """
    xc = float(bbox[0])
    yc = float(bbox[1])
    w = float(bbox[2])
    h = float(bbox[3])

    xmin = int((xc - 0.5*w) * width)
    ymin = int((yc - 0.5*h) * height)
    xmax = int((xc + 0.5*w) * width)
    ymax = int((yc + 0.5*h) * height)

    return [xmin, ymin, xmax, ymax]

# def convert_line(line):


def gather_bboxes(infolder, outfile):
    """Convert the VoTT labels from 
    [x_center, y_center, width, height] to
    [x_min, y_min, x_max, y_max] (also known as
    [x1, y1, x2, y2])
    
    Note the use of hardcoded image name suffixes"""
    # Hardcoded suffixes - modify as needed
    img_files = glob.glob(infolder + os.sep + '*.jpg')
    img_files.extend(glob.glob(infolder + os.sep + '*.JPG'))
    print('Converting labels for {} images'.format(len(img_files)))

    new_lines = []

    for img_file in img_files:
        width, height = get_img_dims(img_file)

        annot_filename = img_file.split('.')[:-1]
        annot_filename = '.'.join(annot_filename) + '.txt'
        print(annot_filename)

        with open(annot_filename, 'r') as f:
            bboxes = f.readlines()
            bboxes_only = [bbox.split(' ')[1:] for bbox in bboxes]
            classes = [bbox.split(' ')[0] for bbox in bboxes]

            # Convert bboxes
            new_bboxes = []
            for bbox in bboxes_only:
                new_bbox = center_to_x1y1x2y2(bbox, width, height)
                new_bboxes.append(new_bbox)

            output_bboxes = []
            for i in range(len(new_bboxes)):
                bbox = [str(x) for x in new_bboxes[i]]
                new_output_box = ','.join([classes[i]] + bbox)
                output_bboxes.append(new_output_box)

            if len(output_bboxes) > 0:
                output_line = ' '.join([img_file] + [' '.join(output_bboxes)]) + '\n'
                new_lines.append(output_line)

    with open(outfile, 'w') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    # Command line options
    parser.add_argument(
        '--annot_folder', type=str,
        help='Annotations folder with individual bounding box label files (and images) in YOLO format'
    )
    parser.add_argument(
        '--outfile', type=str,
        help='Output file name for new formats'
    )

    args = parser.parse_args()

    gather_bboxes(args.annot_folder, args.outfile)



