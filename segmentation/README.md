
## Transferring to Detection

The `train_net.py` script reproduces the semantic segmentation experiments on Cityscapes, the model is deeplabv3+.

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
        $ mkdir -p datasets/cityscapes && cd datasets/cityscapes
        $ ln -s /path/to/cityscapes/gtFine/ .
        $ ln -s /path/to/cityscapes/leftImg8bit/ .
        $ python -m pip install git+https://github.com/mcordts/cityscapesScripts.git
        $ git clone --depth 1 https://github.com/mcordts/cityscapesScripts.git
        $ CITYSCAPES_DATASET=./datasets/cityscapes python cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py
     ```

4. Run training:
   ```
   # r50 
   python train_net.py --config-file configs/deeplab_v3_plus_R_50_os16_mg124_poly_90k_bs16.yaml \
	--num-gpus 8 MODEL.WEIGHTS ./output.pkl
   # r101 
   python train_net.py --config-file configs/deeplab_v3_plus_R_101_os16_mg124_poly_90k_bs16.yaml \
	--num-gpus 8 MODEL.WEIGHTS ./output.pkl
   ```
    
    Or you can see [dist_train.sh](./dist_train.sh) for the training scripts.

