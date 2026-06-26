ROUNDS=(0 1 2 3 4)
AGENT=drivoR
for r in "${ROUNDS[@]}"; do
    TRAIN_TEST_SPLIT=synthetic_reaction_pdm_v1.0-${r}
    CACHE_PATH=${NAVSIM_DEVKIT_ROOT}/cache/${AGENT}_${TRAIN_TEST_SPLIT}_cache

    echo "=== Caching ${TRAIN_TEST_SPLIT} ==="

    python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
        train_test_split=${TRAIN_TEST_SPLIT} \
        agent=$AGENT \
        experiment_name=cache_dataset_${TRAIN_TEST_SPLIT} \
        cache_path=${CACHE_PATH} \
        worker.threads_per_node=10
done
