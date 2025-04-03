#!/bin/bash

model="TransferOT"

if [ _"" = _"${CUDA_VISIBLE_DEVICES}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

# torch DDP settings
nproc_per_node=$NPROC_PER_NODE
nnodes=$NNODES
node_rank=$NODE_RANK
master_addr=$YOUR_ADDR
master_port=$YOUR_PORT

dataset=$DATASET  # ("RealLife" "DOLOS")
sources=$SOURCES  # ("dfew" "ferv39k" "mafw")
backbone=$BACKBONE  # ("VideoMAE" "W2V2_Model" "FusionModel")

# output dir setting
prefix="output"

for source in ${sources[@]};
do
    source_feature="source_domain/${source}_features.hdf5"
    for ds in ${dataset[@]};
    do
        for bb in ${backbone[@]};
        do
            output_dir="${prefix}/backbone_${bb}/tgt_${ds}_src_${source}"
            if [ ! -d "$output_dir" ]; then
                mkdir -p $output_dir
            fi
            torchrun \
                --nproc_per_node=$nproc_per_node \
                --nnodes=$nnodes \
                --node_rank=$node_rank \
                --master_addr=$master_addr \
                --master_port=$master_port \
                train.py \
                --dataset $ds \
                --model $model \
                --backbone_type $bb \
                --output_dir  $output_dir \
                --source_feature $source_feature \
                --distributed
        done
    done
done
echo "Done"