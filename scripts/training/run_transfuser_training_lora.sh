TRAIN_TEST_SPLIT=navtrain
AGENT=transfuser_agent
CACHE_PATH=${NAVSIM_DEVKIT_ROOT}/cache/${AGENT}_${TRAIN_TEST_SPLIT}_cache
# TRAIN_TEST_SPLIT=navmini

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    dataloader.params.batch_size=64 \
    agent=$AGENT \
    agent.config.latent=True \
    agent.config.use_memory=True \
    agent.config.memory_embedding_model=SIGLIP2 \
    agent.config.use_lora=True \
    agent.config.lora_target=memory+decoder \
    agent.lr=1e-4 \
    agent.checkpoint_path=/workspace/MemAD/navsim/download/dataset/models/best_models/transfuser_baseline.ckpt \
    trainer.params.strategy=ddp_find_unused_parameters_true \
    cache_path=$CACHE_PATH \
    experiment_name=transfuser_lora \
    train_test_split=$TRAIN_TEST_SPLIT \
