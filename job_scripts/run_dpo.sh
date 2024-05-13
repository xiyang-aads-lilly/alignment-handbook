echo $PRIMARY:$PRIMARY_PORT:$SLURM_PROCID
echo $TRANSFORMERS_CACHE
echo $HF_DATASETS_CACHE

echo "Primary node: $PRIMARY"
echo "Primary TCP port: $PRIMARY_PORT"

WORLD_SIZE=$(($SLURM_JOB_NUM_NODES*$SLURM_GPUS_PER_TASK))

echo $SLURM_JOB_NUM_NODES $WORLD_SIZE

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file /red/gatortron-phi/workspace/zzz/alignment-handbook/recipes/accelerate_configs/deepspeed_zero2.yaml \
    --main_process_ip $PRIMARY \
    --main_process_port $PRIMARY_PORT \
    --machine_rank $SLURM_PROCID \
    --num_machines $SLURM_JOB_NUM_NODES \
    --num_processes $WORLD_SIZE \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$PRIMARY:$PRIMARY_PORT" \
    --tee 3 \
    /red/gatortron-phi/workspace/zzz/alignment-handbook/scripts/run_dpo.py \
    /red/gatortron-phi/workspace/zzz/alignment-handbook/recipes/zephyr-7b-beta/dpo/config_full.yaml