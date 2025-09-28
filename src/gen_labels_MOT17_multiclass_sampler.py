import os.path as osp
import os
import numpy as np
from random import randint

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

parser.add_argument("--keep_percentage",
                   default=1,
                   type=float,
                   help="Amount of frames kept per sequence")

parser.add_argument("--img_collection_file",
                    default=None,
                    help="Output file with the list of images of the dataset")

args = parser.parse_args()

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)
    else:
        raise \
        FileExistsError(f"Directory already exist, please remove it before continuing.\n -r rm {d}")

def touch_datafile(filename):
    if filename is None:
        print("WARNING: no list of images will be created (img_collection_file).")
        return
    with open(filename, 'w') as _:
        pass

seq_root = args.seq_root_dir
label_root = args.out_dir + f"/labels_with_ids_{int(100 * args.keep_percentage):03d}"
mkdirs(label_root)
touch_datafile(args.img_collection_file)
seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1

for seq in seqs:

    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width =  int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    seq_len_bound = float(seq_info[seq_info.find("seqLength=") + 10:seq_info.find("\nimWidth")])
    seq_len_bound = seq_len_bound * args.keep_percentage
    accepted_fids = -1 * np.ones(int(seq_len_bound), dtype=int)
    
    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
    if 1 == len(gt.shape):  # Sequences that only have one object in the entire
        gt = np.array([gt]) # sequence in one frame. 
    gt = gt[gt[:,1].argsort()]

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)
    img_list_str = "" if args.img_collection_file else None
    
    for fid_str, tid, x, y, w, h, mark, label, _ in gt:
        """ Legacy: only being trained on pedestrians, we abandong this, to
        to train on all classes, The network is capable to classify based on the number of 
        headmap heads it has.
        if mark == 0 or not label == 1:
            continue
        """

        fid = int(fid_str)
        if not(fid in accepted_fids):
            if not np.any(accepted_fids < 0): 
                continue
            
            if randint(0, 1):
                empty_seat = np.where(accepted_fids == -1)
                accepted_fids[empty_seat[0][0]] = fid
                img_list_str = img_list_str + f"{seq_root}/{seq}/img1/{fid}.jpg\n" \
                    if not (img_list_str is None) \
                    else None
            else:
                continue

        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label = int(label) - 1

        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            label, tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)

    if img_list_str is None:
        continue

    with open(args.img_collection_file, 'a') as f:
        f.write(img_list_str)

