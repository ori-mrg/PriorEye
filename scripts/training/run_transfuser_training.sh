TRAIN_TEST_SPLIT=navtrain
AGENT=transfuser_agent
CACHE_PATH=${NAVSIM_DEVKIT_ROOT}/cache/${AGENT}_${TRAIN_TEST_SPLIT}_cache
# TRAIN_TEST_SPLIT=navmini

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=$AGENT \
    agent.config.latent=True \
    agent.config.use_memory=True \
    agent.config.memory_embedding_model=SIGLIP2 \
    cache_path=$CACHE_PATH \
    experiment_name=transfuser_memory_v2 \
    train_test_split=$TRAIN_TEST_SPLIT \
    dataloader.params.batch_size=16