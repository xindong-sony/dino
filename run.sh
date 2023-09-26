NODE_RANK=$1
export NCCL_SOCKET_IFNAME=ens3
export NCCL_DEBUG=ALL
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG_SUBSYS=ALL

echo $NODE_RANK

python -m torch.distributed.launch --nproc_per_node=4 \
           --nnodes=2 --node_rank=$NODE_RANK --master_addr="10.4.27.83" \
           --master_port=50000 main_dino.py --arch swin_t \
           --data_path /home/ubuntu/firstbatch_syn/ \
           --num_workers 6 --batch_size_per_gpu 32\
           --output_dir /home/ubuntu/dino/on_firstbatch_syn \
           --dist_url tcp://10.4.27.83:50000


# python -m torch.distributed.launch --nproc_per_node=4 \
#             main_dino.py --arch vit_small \
#             --data_path /home/ubuntu/firstbatch_syn/ \
#             --output_dir /home/ubuntu/dino/on_firstbatch_syn
