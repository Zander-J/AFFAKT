#!/bin/bash

model="TransferOT"

# torch DDP settings
if [ _"" = _"${CUDA_VISIBLE_DEVICES}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

nproc_per_node=1
nnodes=1
node_rank=0
master_addr="127.0.0.1"
master_port=36559

dataset=("RealLife")  # ("RealLife" "DOLOS")
sources=("dfew")  # ("dfew" "ferv39k" "mafw")
backbone=("VideoMAE")  # ("VideoMAE" "W2V2_Model" "FusionModel")
seed=3407
num_epochs=20
batch_size=4
lr=1e-5

# output dir setting
general_output="output"
version_method="transfer_ot"
prefix="$general_output/$version_method/model_$model"

deltas=(0.01)
xi=0.2
nu=0.05
alpha=0.95

for source in ${sources[@]};
do
    source_feature="source_domain/${source}_features.hdf5"
    for ds in ${dataset[@]};
    do
        for bb in ${backbone[@]};
        do
            output_dir="${prefix}/backbone_${bb}/tgt_${ds}_src_${source}_seed_${seed}_lr_${lr}_epc_${num_epochs}_bs_${batch_size}"
            if [ ! -d "$output_dir" ]; then
                mkdir -p $output_dir
            fi
            for delta in ${deltas[@]}
            do
                torchrun \
                    --nproc_per_node=$nproc_per_node \
                    --nnodes=$nnodes \
                    --node_rank=$node_rank \
                    --master_addr=$master_addr \
                    --master_port=$master_port \
                    train.py \
                    --seed $seed \
                    --dataset $ds \
                    --xi $xi \
                    --nu $nu \
                    --delta $delta \
                    --alpha $alpha \
                    --num_epochs $num_epochs \
                    --model TransferOT \
                    --backbone_type $bb \
                    --lr $lr \
                    --batch_size $batch_size \
                    --output_dir  $output_dir \
                    --source_feature $source_feature \
                    --amp \
                    --adapter \
                    --memory_sufficient \
                    --distributed \
                    --write_to_local \
                    --git
            done
        done
    done
done
echo "Done"