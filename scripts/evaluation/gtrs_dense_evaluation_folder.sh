
export PROGRESS_MODE="eval"
TRAIN_TEST_SPLIT=navhard_two_stage
CACHE_PATH=${NAVSIM_DEVKIT_ROOT}/cache/${TRAIN_TEST_SPLIT}_metric_cache

# Default path configuration
CKPT_DIR="/dataset/models/gtrs_baseline/ablation_map_vector"
export DP_PREDS=${NAVSIM_DEVKIT_ROOT}/traj_final/dp_baseline_subscore_navhard_two_stage.pkl;


for CHECKPOINT in ${CKPT_DIR}/*.ckpt; do
    CKPT_NAME=$(basename "$CHECKPOINT" .ckpt)
    EPOCH=$(echo "$CKPT_NAME" | cut -d'-' -f1)

    if [ "$EPOCH" -lt 10 ] || [ "$EPOCH" -ge 30 ]; then
        echo "Skipping epoch: $EPOCH (only processing 7 <= epoch < 12)"
        continue
    fi

    echo "----------------------------------------------------------"
    echo "Processing checkpoint: $CKPT_NAME"
    echo "----------------------------------------------------------"

    EXP_NAME="gtrs_dense_baseline_map_vector_${TRAIN_TEST_SPLIT}_${CKPT_NAME}"

    python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2.py \
        dataloader.params.batch_size=32 \
        agent=gtrs_dense_vov \
        +combined_inference=true \
        agent.config.use_memory=True \
        agent.config.memory_embedding_model=MAP_VECTOR \
        agent.checkpoint_path=$CHECKPOINT \
        agent.config.vocab_path=${NAVSIM_DEVKIT_ROOT}/traj_final/8192.npy \
        trainer.params.precision=16-mixed \
        experiment_name=$EXP_NAME \
        +cache_path=null \
        metric_cache_path=$CACHE_PATH \
        train_test_split=$TRAIN_TEST_SPLIT

done

