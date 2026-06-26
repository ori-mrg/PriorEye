NUM_NODES=1
MASTER_ADDR=127.0.0.1 # your master node ip
NODE_RANK=0 # 0 for the master node, 1 and 2 for other sub-nodes
config="competition_training" # this config uses the entire navtrain dataset for training
TRAIN_TEST_SPLIT=navtrain
export AGENT=drivoR
CACHE_PATH=${NAVSIM_DEVKIT_ROOT}/cache/${AGENT}_${TRAIN_TEST_SPLIT}_cache
experiment_name=drivoR_symscale

# training hyper-parameters
bs=12 # 16

export SYN_IDX="0,1,2,3,4"
export SYN_GT=pdm

MASTER_PORT=29501 MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${NUM_NODES} NODE_RANK=${NODE_RANK} \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_drivoR.py  \
    --config-name ${config} \
    agent=$AGENT \
    +train_ckpt_path=null \
    +validation_run=false \
    experiment_name=${experiment_name} \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path=$CACHE_PATH \
    use_cache_without_dataset=true \
    trainer.params.max_epochs=30 \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size=${bs} \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=4 \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 \
    agent.config.long_trajectory_additional_poses=2 \
    agent.config.use_memory=true \
    agent.config.memory_embedding_model=SIGLIP2 \
    seed=2
