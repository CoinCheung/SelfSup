
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# for pretrain
ARCH=resnet50
# ARCH=resnet101
DATAPATH=/data/zzy/.datasets/imagenet
# DATAPATH=/data/rdisk/imagenet
# # DATAPATH=$1
# # URL='tcp://localhost:10001'
URL='tcp://10.128.61.6:20004'
WORD_SIZE=4
RANK=2
EPOCHS=200
LR=0.12 # use 4 nodes
BATCHSIZE=256 # bs of 1 node
time python main_pretrain.py -a $ARCH --lr $LR --batch-size $BATCHSIZE --epochs $EPOCHS --world-size $WORD_SIZE --rank $RANK --dist-url $URL --multiprocessing-distributed --use-mixed-precision --mlp --moco-t 0.2 --aug-plus --cos --fast-moco \
$DATAPATH

# linear eval and finetune
URL='tcp://localhost:20021'
WORD_SIZE=1
RANK=0
DATAPATH=/data/zzy/.datasets/imagenet/
PRETRAINED=./checkpoint_0199.pth.tar
ARCH=resnet50
LR=240.0
BS=2048

# linear eval
# python main_finetune.py -a $ARCH --lr $LR --batch-size $BS --pretrained $PRETRAINED --dist-url $URL --multiprocessing-distributed --world-size $WORD_SIZE --rank $RANK --linear-eval $DATAPATH

# finetune
# python main_finetune.py -a $ARCH --lr 0.4 --weight-decay 0.0001 --batch-size 1024 --cos --pretrained $PRETRAINED --dist-url $URL --multiprocessing-distributed --world-size $WORD_SIZE --rank $RANK $DATAPATH

