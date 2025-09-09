cd src
python train.py mot \
  --exp_id kitti_multiclass_dla34 \
  --load_model "../../../../detectors/fairmot/ctdet_coco_dla_2x.pth" \
  --data_cfg "../src/lib/cfg/kitti_multiclass.json" \
  --gpus 1 \
  --reference_model "../../../../detectors/fairmot/dla34-ba72cf86.pth" \
  --num_classes 9 \
  --print_iter 20 \
  --batch_size 2
cd ..
