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
    gt = open(gt_txt,'r')

    seq_label_root = osp.join(label_root, seq)
    mkdirs(seq_label_root)

    for attr_str in gt:
        attr = attr_str.split(' ')
        fid = int(attr[0])
        tid = int(attr[1]) # attr: 2, 3 and 4 ignored.

        x = float(attr[5]) # left
        y = float(attr[6]) # top
        right = float(attr[7])
        bottom = float(attr[8])
        
        w = right - x
        h = bottom - y
        
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)

    gt.close()
