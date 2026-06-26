TEAM_NAME="P"
AUTHORS="Anonymous"
EMAIL="anonymous@email.com"
INSTITUTION="Anonymous"
COUNTRY="Anonymous"
TRAIN_TEST_SPLIT=navhard_two_stage
CHECKPOINT=/workspace/MemAD/navsim/download/dataset/models/best_models/gtrs_dense_memory.ckpt
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

EXP_NAME=gtrs_dense_memory_submission_${TRAIN_TEST_SPLIT}

export DP_PREDS=${NAVSIM_DEVKIT_ROOT}/traj_final/dp_baseline_subscore_navhard_two_stage.pkl

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle_gpu.py \
    dataloader.params.batch_size=48 \
    agent=gtrs_dense_vov \
    +combined_inference=true \
    agent.config.use_memory=True \
    agent.config.memory_embedding_model=SIGLIP2 \
    agent.checkpoint_path=$CHECKPOINT \
    agent.config.vocab_path=${NAVSIM_DEVKIT_ROOT}/traj_final/8192.npy \
    trainer.params.precision=16-mixed \
    experiment_name=$EXP_NAME \
    +cache_path=null \
    train_test_split=$TRAIN_TEST_SPLIT \
    synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
    synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
    team_name=$TEAM_NAME \
    authors=$AUTHORS \
    email=$EMAIL \
    institution=$INSTITUTION \
    country=$COUNTRY \
