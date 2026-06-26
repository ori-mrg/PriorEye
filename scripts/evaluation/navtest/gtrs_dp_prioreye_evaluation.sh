
export PROGRESS_MODE="eval"
TRAIN_TEST_SPLIT=navtest
CHECKPOINT=${NAVSIM_DEVKIT_ROOT}/models/dp_prioreye.ckpt
CACHE_PATH=${NAVSIM_DEVKIT_ROOT}/cache/${TRAIN_TEST_SPLIT}_metric_cache
agent=gtrs_diffusion_policy
subscore_folder=dp_subscore_prioreye_${TRAIN_TEST_SPLIT}
experiment_name=dp_prioreye_${TRAIN_TEST_SPLIT}

export DP_PREDS=none
export SUBSCORE_PATH=${NAVSIM_EXP_ROOT}/${subscore_folder}/dp_prioreye_subscore_${TRAIN_TEST_SPLIT}.pkl # save path for the dp-generated trajectories

mkdir -p ${NAVSIM_EXP_ROOT}/${subscore_folder}

python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage_gpu.py \
    traffic_agents=reactive \
    agent=$agent \
    agent.config.use_memory=true \
    agent.config.memory_embedding_model=SIGLIP2 \
    dataloader.params.batch_size=48 \
    dataloader.params.num_workers=8 \
    agent.checkpoint_path=$CHECKPOINT \
    trainer.params.precision=32 \
    experiment_name=${experiment_name} \
    +cache_path=null \
    metric_cache_path=${CACHE_PATH} \
    train_test_split=$TRAIN_TEST_SPLIT \
