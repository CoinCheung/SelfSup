
export GLOO_SOCKET_IFNAME=eth0

CKPT=../checkpoint_0199.pth.tar
CONFIG=configs/deeplab_v3_plus_R_50_os16_mg124_poly_90k_bs16.yaml
# CONFIG=configs/deeplab_v3_plus_R_101_os16_mg124_poly_90k_bs16.yaml


rm ./output_ckpt_200ep.pkl
python convert-pretrain-to-detectron2.py ../checkpoint_0199.pth.tar ./output_ckpt_200ep.pkl


python train_net.py --config-file $CONFIG --num-gpus 8 MODEL.WEIGHTS ./output_ckpt_200ep.pkl
