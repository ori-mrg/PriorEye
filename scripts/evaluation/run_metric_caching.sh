# TRAIN_TEST_SPLIT=navtest
TRAIN_TEST_SPLIT=navhard_two_stage

CACHE_PATH=${NAVSIM_CACHE_ROOT}/${TRAIN_TEST_SPLIT}_metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
metric_cache_path=$CACHE_PATH 