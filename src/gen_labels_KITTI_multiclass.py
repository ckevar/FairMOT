import os.path as osp
import os
import numpy as np

import argparse
from PIL import Image

parser = argparse.ArgumentParser()
# basic experiment setting
parser.add_argument("--seq_root_dir", 
                    default=None, 
                    help='Path to KITTI format.',
                    required=True)
 
parser.add_argument("--out_dir",
                    default=None,
                    help="Path to output dir",
                    required=True)

args = parser.parse_args()

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

KITTI_labels = {
        'Car': 1,
        'Van': 2,
        'Truck': 3,
        'Pedestrian': 4, 
        'Person': 5, # person sitting
        'Cyclist': 6, 
        'Tram': 7,
        'Misc': 8,
        'DontCare': 9
        }

def kitti_np_loadtxt(filename):
    """ This only reads the attributes of interest for FairMOT:
    1. frame id
    2. track id
    3. class 
    4. box left
    5. box top
    6. box right
    7. box bottom
    """
    gt = open(filename, 'r')

    sequence = [[], [], [], [], [], [], []]
    for attr_str in gt:
        attr = attr_str.split(' ')
        sequence[0].append(float(attr[0])) # It should be int, but when the lists
        sequence[1].append(float(attr[1])) # get cast into arrays, what we got is
        sequence[2].append(float(KITTI_labels[attr[2]])) # array of floats.
        sequence[3].append(float(attr[6]))
        sequence[4].append(float(attr[7]))
        sequence[5].append(float(attr[8]))
        sequence[6].append(float(attr[9]))

    gt.close()

    sequence = np.array(sequence).T
    # Sort by track ids
    sequence = sequence[sequence[:, 1].argsort()]

    return sequence



seq_root = args.seq_root_dir
label_root = args.out_dir
mkdirs(label_root)
img_dir = "/image_02"
label_dir = "/label_02"
seqs = [s for s in os.listdir(seq_root + img_dir)]

tid_curr = 0
tid_last = -1
for seq in seqs:
    img_seq_dir = f"{seq_root}/{img_dir}/{seq}"

    img_sample = os.listdir(img_seq_dir)[0]
    img = Image.open(f"{img_seq_dir}/{img_sample}")
    seq_width, seq_height = img.size
    img.close()

    gt_txt = f"{seq_root}/{label_dir}/{seq}.txt"
    gt = kitti_np_loadtxt(gt_txt)

    seq_label_root = osp.join(label_root, seq)
    mkdirs(seq_label_root)

    for attr in gt:
        fid = int(attr[0])
        tid = int(attr[1])
        cls = int(attr[2]) - 1
        x = attr[3] # left
        y = attr[4] # top
        right = attr[5]
        bottom = attr[6]
        
        w = right - x
        h = bottom - y
        
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            cls, tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)

        with open(label_fpath, 'a') as f:
            f.write(label_str)

