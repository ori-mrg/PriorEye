TRAIN_TEST_SPLIT=navhard_two_stage

CHECKPOINT=${NAVSIM_DEVKIT_ROOT}/models/gtrs_dense_prioreye.ckpt
CACHE_PATH=${NAVSIM_DEVKIT_ROOT}/cache/${TRAIN_TEST_SPLIT}_metric_cache


EXP_NAME=gtrs_dense_prioreye_${TRAIN_TEST_SPLIT}
export PROGRESS_MODE="eval"

# export DP_PREDS=/workspace/MemAD/exp/dp_subscore_navhard_two_stage/dp_navhard_two_stage.pkl;
export DP_PREDS=${NAVSIM_DEVKIT_ROOT}/traj_final/dp_prioreye_subscore_navhard_two_stage.pkl;

dir=gtrs_dense_prioreye_subscore
export SUBSCORE_PATH=${NAVSIM_EXP_ROOT}/${dir}/gtrs_dense_baseline_${TRAIN_TEST_SPLIT}.pkl # save path for the dp-generated trajectories
mkdir -p ${NAVSIM_EXP_ROOT}/${dir}


python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2.py \
    dataloader.params.batch_size=16 \
    agent=gtrs_dense_vov \
    +combined_inference=true \
    agent.config.use_memory=True \
    agent.config.memory_embedding_model=SIGLIP2 \
    agent.checkpoint_path=$CHECKPOINT \
    agent.config.vocab_path=${NAVSIM_DEVKIT_ROOT}/traj_final/8192.npy \
    trainer.params.precision=16-mixed \
    experiment_name=$EXP_NAME \
    +cache_path=null \
    metric_cache_path=$CACHE_PATH \
    train_test_split=$TRAIN_TEST_SPLIT \
