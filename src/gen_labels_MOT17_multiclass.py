import os.path as osp
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser()
# basic experiment setting
parser.add_argument("--seq_root_dir", 
                    default=None, 
                    help='Path to MOT17 format.',
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
label_root = args.out_dir + "/train"
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1
for seq in seqs:
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
    if 1 == len(gt.shape):  # Sequences that only have one object in the entire
        gt = np.array([gt]) # sequence in one frame. 

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)
    
    for fid, tid, x, y, w, h, mark, label, _ in gt:
        """ Legacy: only being trained on pedestrians, we abandong this, to
        to train on all classes, but keep in mind that this network DOES NOT
        classify (the network DOES NOT yield a class).
        if mark == 0 or not label == 1:
            continue
        """
        fid = int(fid)
        tid = int(tid)
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
