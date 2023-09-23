NODE_RANK=$1
export NCCL_SOCKET_IFNAME=ens3
export NCCL_DEBUG=ALL

echo $NODE_RANK

python -m torch.distributed.launch --nproc_per_node=4 \
           --nnodes=2 --node_rank=$NODE_RANK --master_addr="10.4.27.83" \
           --master_port=12345 main_dino.py --arch vit_small \
           --data_path /home/ubuntu/firstbatch_syn/images/ \
           --output_dir /home/ubuntu/dino/on_firstbatch_syn \
           --dist_url tcp://10.4.27.83:12345