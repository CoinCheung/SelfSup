
## Transferring to Detection

The `train_net.py` script reproduces the object detection experiments on Pascal VOC and COCO.

### Instruction

1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).
    ```
    $ git clone https://github.com/facebookresearch/detectron2.git
    $ cd detectron2
    $ git checkout 48b598b4f61fbb24
    $ python -m pip install -e .
    ```
    This requires cuda11.3 to work.

2. Convert a pre-trained model to detectron2's format:
   ```
   python3 convert-pretrain-to-detectron2.py input.pth.tar output.pkl
   ```

3. Put dataset under "./datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.
     ```
        $ mkdir -p datasets/coco && cd datasets/coco
        $ ln -s /path/to/coco/train2017/ .
        $ ln -s /path/to/coco/val2027/ .
        $ ln -s /path/to/coco/annotations/ . # this is unzipped from annotations.zip
     ```

4. Run training:
   ```
   # r50 
   python train_net.py --config-file configs/coco_R_50_C4_2x_moco.yaml \
	--num-gpus 8 MODEL.WEIGHTS ./output.pkl
   # r101 
   python train_net.py --config-file configs/coco_R_101_C4_2x_moco.yaml \
	--num-gpus 8 MODEL.WEIGHTS ./output.pkl
   ```
    
    Or you can see [dist_train.sh](./dist_train.sh) for the training scripts.

