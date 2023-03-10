
## detection on coco
pushd detection/
export GLOO_SOCKET_IFNAME=eth0

CONFIG=configs/coco_R_50_FPN_1x_moco.yaml
NGPUS=8

CKPT=/nfs-data/zzy/.dataset/SparK/_nfs_data_zzy__dataset_SparK_output_org_impl/r50_org_state_final.pth
python train_net.py --config-file $CONFIG --num-gpus $NGPUS MODEL.WEIGHTS $CKPT 2>&1 | tee log_det_spark_org_$1.txt

sleep 120

CKPT=/nfs-data/zzy/.dataset/SparK/_nfs_data_zzy__dataset_SparK_output_my_impl/r50_my_state_final.pth
python train_net.py --config-file $CONFIG --num-gpus $NGPUS MODEL.WEIGHTS $CKPT 2>&1 | tee log_det_spark_my_$1.txt

popd


sleep 120

## semantic segmentation on city
pushd segmentation/
export GLOO_SOCKET_IFNAME=eth0

CONFIG=configs/deeplab_v3_plus_R_50_os16_mg124_poly_90k_bs16.yaml
NGPUS=8
PORT=44333


CKPT=/nfs-data/zzy/.dataset/SparK/_nfs_data_zzy__dataset_SparK_output_org_impl/r50_org_state_final.pth
python train_net.py --config-file $CONFIG --num-gpus $NGPUS --dist-url 'tcp://127.0.0.1:'$PORT  MODEL.WEIGHTS $CKPT 2>&1 | tee log_seg_spark_org_$1.txt

sleep 120

CKPT=/nfs-data/zzy/.dataset/SparK/_nfs_data_zzy__dataset_SparK_output_my_impl/r50_my_state_final.pth
python train_net.py --config-file $CONFIG --num-gpus $NGPUS --dist-url 'tcp://127.0.0.1:'$PORT  MODEL.WEIGHTS $CKPT 2>&1 | tee log_seg_spark_my_$1.txt

popd
