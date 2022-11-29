
export GLOO_SOCKET_IFNAME=eth0

CKPT=../checkpoint_0199.pth.tar
CONFIG=configs/deeplab_v3_plus_R_50_os16_mg124_poly_90k_bs16.yaml
# CONFIG=configs/deeplab_v3_plus_R_101_os16_mg124_poly_90k_bs16.yaml


rm ./output_ckpt_200ep.pkl
python convert-pretrain-to-detectron2.py ../checkpoint_0199.pth.tar ./output_ckpt_200ep.pkl

# python convert-pretrain-to-detectron2.py ../pretrained/r50_checkpoint_0199_mocov2_cutmix.pth.tar ./output_ckpt_200ep.pkl


# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# NGPUS=8
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,4
NGPUS=4
PORT=44332
python train_net.py --config-file $CONFIG --num-gpus $NGPUS --dist-url 'tcp://127.0.0.1:'$PORT  MODEL.WEIGHTS ./output_ckpt_200ep.pkl
