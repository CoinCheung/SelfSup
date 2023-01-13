
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

rm log_linear_6127*txt log_finetune_6127*txt detection/log_det_6127*txt

# for pretrain
ARCH=resnet18
# ARCH=resnet101
DATAPATH=/data/zzy/.datasets/imagenet
URL='tcp://127.0.0.1:20006'
WORD_SIZE=1
RANK=0
EPOCHS=200
LR=0.12 # use 1 nodes, (1 x 1024 / 256) x 0.03 = 0.12
BATCHSIZE=1024 # bs of 1 node
time python main_pretrain.py -j 64 --ckpt_prefix r18 -a $ARCH --lr $LR --batch-size $BATCHSIZE --epochs $EPOCHS --world-size $WORD_SIZE --rank $RANK --dist-url $URL --multiprocessing-distributed --use-mixed-precision --mlp --moco-t 0.2 --aug-plus --cos --fast-moco --cutmix --mixup --dense $DATAPATH

# # linear eval and finetune
# URL='tcp://localhost:20021'
# WORD_SIZE=1
# RANK=0
# DATAPATH=/data/zzy/.datasets/imagenet/
# PRETRAINED=./checkpoint_0199.pth.tar
# ARCH=resnet50
# LR=240.0
# BS=2048
#
# sleep 120
#
# # linear eval
# for i in $(seq 1 1 4);
# do
# time python main_finetune.py -a $ARCH --lr $LR --batch-size $BS --pretrained $PRETRAINED --dist-url $URL --multiprocessing-distributed --world-size $WORD_SIZE --rank $RANK --linear-eval $DATAPATH 2>&1 | tee log_linear_6127_$i.txt
# done
#
# sleep 120
#
# # finetune
# for i in $(seq 1 1 4);
# do
# python main_finetune.py -a $ARCH --lr 0.4 --weight-decay 0.0001 --batch-size 1024 --cos --pretrained $PRETRAINED --dist-url $URL --multiprocessing-distributed --world-size $WORD_SIZE --rank $RANK $DATAPATH 2>&1 | tee log_finetune_6127_$i.txt
# done
#
#
# sleep 120
#
#
# ## detection on coco
# pushd detection/
# for i in $(seq 1 1 4);
# do
# bash dist_train.sh 2>&1 | tee log_det_6127_$i.txt
# done
# popd



## finetune scratch
# python main_finetune.py -a $ARCH --lr 0.4 --weight-decay 0.0001 --batch-size 1024 --cos --dist-url $URL --multiprocessing-distributed --world-size $WORD_SIZE --rank $RANK $DATAPATH

