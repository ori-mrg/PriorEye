NUM_NODES=1
MASTER_ADDR=127.0.0.1 # your master node ip
NODE_RANK=0 # 0 for the master node, 1 and 2 for other sub-nodes
config="competition_training" # this config uses the entire navtrain dataset for training
TRAIN_TEST_SPLIT=navtrain
AGENT=gtrs_diffusion_policy
CACHE_PATH=${NAVSIM_DEVKIT_ROOT}/cache/${AGENT}_${TRAIN_TEST_SPLIT}_cache
experiment_name=dp_baseline

# training hyper-parameters
lr=0.0002
bs=16
max_epochs=80

MASTER_PORT=29500 MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${NUM_NODES} NODE_RANK=${NODE_RANK} \
        python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_dense.py \
            --config-name ${config} \
            trainer.params.num_nodes=${NUM_NODES} \
            agent=${AGENT} \
            experiment_name=${experiment_name} \
            train_test_split=$TRAIN_TEST_SPLIT \
            dataloader.params.batch_size=${bs} \
            ~trainer.params.strategy \
            trainer.params.max_epochs=${max_epochs} \
            trainer.params.precision=16-mixed \
            agent.config.ckpt_path=${experiment_name} \
            agent.lr=${lr} \
            cache_path=${CACHE_PATH}