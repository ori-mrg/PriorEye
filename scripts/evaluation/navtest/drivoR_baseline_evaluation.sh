TRAIN_TEST_SPLIT=navtest

CHECKPOINT=${NAVSIM_DEVKIT_ROOT}/models/drivoR_baseline.ckpt
CACHE_PATH=${NAVSIM_DEVKIT_ROOT}/cache/${TRAIN_TEST_SPLIT}_metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/${TRAIN_TEST_SPLIT}/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/${TRAIN_TEST_SPLIT}/synthetic_scene_pickles

EXP_NAME=drivoR_baseline_${TRAIN_TEST_SPLIT}
export PROGRESS_MODE="eval"


AGENT=drivoR
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage_gpu.py  \
    traffic_agents=reactive \
    train_test_split=$TRAIN_TEST_SPLIT \
    experiment_name=$EXP_NAME \
    metric_cache_path=$CACHE_PATH \
    synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
    synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
    agent=$AGENT \
    agent.checkpoint_path=$CHECKPOINT \
    agent.config.proposal_num=64 \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.config.noc=10 \
    agent.config.dac=13 \
    agent.config.ddc=6 \
    agent.config.ttc=14 \
    agent.config.ep=15 \
    agent.config.comfort=2 \
    agent.config.use_memory=false \
    agent.config.memory_embedding_model=SIGLIP2 \
