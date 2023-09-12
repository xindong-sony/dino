NODE_RANK=$1
export NCCL_SOCKET_IFNAME=ens3
python -m torch.distributed.launch --nproc-per-node=4
           --nnodes=2 --node-rank=$NODE_RANK --master-addr="10.4.19.86"
           --master-port=1234 main_dino.py --arch vit_small --data_path /home/ubuntu/imagenet_dataset/imagenet/train --output_dir /home/ubuntu/