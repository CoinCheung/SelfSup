
export GLOO_SOCKET_IFNAME=eth0

CKPT=../checkpoint_0199.pth.tar
# CONFIG=configs/pascal_voc_R_50_C4_24k_moco.yaml
CONFIG=configs/coco_R_50_C4_1x_moco.yaml
# CONFIG=configs/coco_R_50_FPN_1x_moco.yaml
# CONFIG=configs/coco_R_101_C4_1x_moco.yaml

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# NGPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NGPUS=8

rm ./output_ckpt_200ep.pkl
python convert-pretrain-to-detectron2.py $CKPT ./output_ckpt_200ep.pkl

# python convert-pretrain-to-detectron2.py ../pretrained/r50_checkpoint_0199_model_15.pth.tar ./output_ckpt_200ep.pkl

python train_net.py --config-file $CONFIG --num-gpus $NGPUS MODEL.WEIGHTS ./output_ckpt_200ep.pkl

## original supervised
# python train_net.py --config-file configs/pascal_voc_R_50_C4_24k.yaml --num-gpus 8
