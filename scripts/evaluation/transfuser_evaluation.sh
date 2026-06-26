TRAIN_TEST_SPLIT=navhard_two_stage
CHECKPOINT=${NAVSIM_DEVKIT_ROOT}/models/transfuser_prioreye.ckpt

CACHE_PATH=${NAVSIM_DEVKIT_ROOT}/cache/${TRAIN_TEST_SPLIT}_metric_cache


SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/${TRAIN_TEST_SPLIT}/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/${TRAIN_TEST_SPLIT}/synthetic_scene_pickles


python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=transfuser_agent \
    worker=ray_distributed \
    agent.checkpoint_path=$CHECKPOINT \
    agent.config.latent=True \
    agent.config.use_memory=True \
    agent.config.memory_embedding_model=SIGLIP2 \
    experiment_name=ltf_memory_navhard_two_stage \
    metric_cache_path=$CACHE_PATH \
    synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
    synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
# train_test_split.scene_filter.max_scenes=100
