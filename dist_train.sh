
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

rm log_linear_$1.txt log_finetune_$1.txt detection/log_det_$1.txt segmentation/log_seg_$1.txt

# for pretrain
ARCH=resnet50
# ARCH=resnet101
DATAPATH=/data/zzy/.datasets/imagenet
URL='tcp://10.128.61.6:20004'
WORD_SIZE=$2
RANK=$1
EPOCHS=200
BATCHSIZE=256 # bs of 1 node

# OPT=SGD
# LR=0.12 # use 4 nodes, 4 x 256 = 1024
# WD=1e-4
OPT=AdamW
LR=8e-4 # use 4 nodes, 4 x 256 = 1024
WD=8e-2

time python main_pretrain.py -a $ARCH --optim $OPT --lr $LR --wd $WD --batch-size $BATCHSIZE --epochs $EPOCHS --world-size $WORD_SIZE --rank $RANK --dist-url $URL --multiprocessing-distributed --use-mixed-precision --mlp --moco-t 0.2 --aug-plus --cos --mae \
    $DATAPATH
    # --fast-moco --cutmix --mixup --dense \

# linear eval and finetune
URL='tcp://localhost:20021'
WORD_SIZE=1
RANK=0
DATAPATH=/data/zzy/.datasets/imagenet/
PRETRAINED=./checkpoint_0199.pth.tar
ARCH=resnet50
LR=240.0
BS=2048

sleep 120

# linear eval
time python main_finetune.py -a $ARCH --lr $LR --batch-size $BS --pretrained $PRETRAINED --dist-url $URL --multiprocessing-distributed --world-size $WORD_SIZE --rank $RANK --linear-eval $DATAPATH 2>&1 | tee log_linear_$1.txt

# sleep 120
#
# # finetune
# python main_finetune.py -a $ARCH --lr 0.4 --weight-decay 0.0001 --batch-size 1024 --cos --pretrained $PRETRAINED --dist-url $URL --multiprocessing-distributed --world-size $WORD_SIZE --rank $RANK $DATAPATH 2>&1 | tee log_finetune_$1.txt


sleep 120


## detection on coco
pushd detection/
bash dist_train.sh 2>&1 | tee log_det_$1.txt
popd


sleep 120

## semantic segmentation on city
pushd segmentation/
bash dist_train.sh 2>&1 | tee log_seg_$1.txt
popd

## finetune scratch
# python main_finetune.py -a $ARCH --lr 0.4 --weight-decay 0.0001 --batch-size 1024 --cos --dist-url $URL --multiprocessing-distributed --world-size $WORD_SIZE --rank $RANK $DATAPATH

