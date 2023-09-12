NODE_RANK=$1
export NCCL_SOCKET_IFNAME=ens3
# export MASTER_ADDR=10.4.19.86
# export MASTER_PORT=1234
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO

python -m torch.distributed.launch --nproc_per_node=4 \
           --nnodes=2 --node_rank=$NODE_RANK --master_addr="10.4.19.86" \
           --master_port=1234 main_dino.py --arch vit_small \
           --data_path /home/ubuntu/imagenet_dataset/imagenet/train \
           --output_dir /home/ubuntu/