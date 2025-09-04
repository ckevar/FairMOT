cd src
python train.py mot \
  --exp_id mot17_dla34 \
  --load_model "../models/ctdet_coco_dla_2x.pth" \
  --data_cfg "../src/lib/cfg/mot17_custom.json" \
  --gpus 1 \
  --reference_model "../../../../detectors/fairmot/dla34-ba72cf86.pth"
cd ..
