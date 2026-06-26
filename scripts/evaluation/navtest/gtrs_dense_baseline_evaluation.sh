TRAIN_TEST_SPLIT=navtest

CHECKPOINT=${NAVSIM_DEVKIT_ROOT}/models/gtrs_dense_baseline.ckpt
CACHE_PATH=${NAVSIM_DEVKIT_ROOT}/cache/${TRAIN_TEST_SPLIT}_metric_cache


EXP_NAME=gtrs_dense_baseline_${TRAIN_TEST_SPLIT}
export PROGRESS_MODE="eval"

# export DP_PREDS=/workspace/MemAD/exp/dp_subscore_navhard_two_stage/dp_navhard_two_stage.pkl;
export DP_PREDS=${NAVSIM_DEVKIT_ROOT}/traj_final/dp_baseline_subscore_navtest.pkl;



python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage_gpu.py \
    traffic_agents=reactive \
    dataloader.params.batch_size=16 \
    agent=gtrs_dense_vov \
    +combined_inference=true \
    agent.config.use_memory=false \
    agent.config.memory_embedding_model=SIGLIP2 \
    agent.checkpoint_path=$CHECKPOINT \
    agent.config.vocab_path=${NAVSIM_DEVKIT_ROOT}/traj_final/8192.npy \
    trainer.params.precision=16-mixed \
    experiment_name=$EXP_NAME \
    +cache_path=null \
    metric_cache_path=$CACHE_PATH \
    train_test_split=$TRAIN_TEST_SPLIT \
