TRAIN_TEST_SPLIT=navtest
CHECKPOINT=${NAVSIM_DEVKIT_ROOT}/models/transfuser_prioreye.ckpt

CACHE_PATH=${NAVSIM_DEVKIT_ROOT}/cache/${TRAIN_TEST_SPLIT}_metric_cache


SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/${TRAIN_TEST_SPLIT}/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/${TRAIN_TEST_SPLIT}/synthetic_scene_pickles




EXP_NAME=transfuser_prioreye_${TRAIN_TEST_SPLIT}
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage_gpu.py \
    traffic_agents=reactive \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=transfuser_agent \
    worker=ray_distributed \
    agent.checkpoint_path=$CHECKPOINT \
    agent.config.latent=True \
    agent.config.use_memory=True \
    agent.config.memory_embedding_model=SIGLIP2 \
    experiment_name=$EXP_NAME \
    metric_cache_path=$CACHE_PATH \
    synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
    synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
# train_test_split.scene_filter.max_scenes=100
