TRAIN_TEST_SPLIT=navtrain
AGENT=transfuser_agent
CACHE_PATH=${NAVSIM_DEVKIT_ROOT}/cache/${AGENT}_${TRAIN_TEST_SPLIT}_cache

PYTHONWARNINGS="ignore:invalid value encountered in cast" \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
train_test_split=$TRAIN_TEST_SPLIT  \
cache_path=$CACHE_PATH \
experiment_name=${TRAIN_TEST_SPLIT}_cache_exp \
agent=$AGENT \
agent.config.latent=True \
# worker=sequential