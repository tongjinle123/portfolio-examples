#!/bin/bash
export POPLAR_TARGET_OPTIONS='{"gatewayMode":"true"}'
poprun -vv --host $HOSTS \
        --vipu-partition=$PARTITION \
        --num-instances=$NUM_INSTANCE --num-replicas=$NUM_REPLICA \
        --num-ilds $NUM_ILDS \
        --ipus-per-replica=8 \
        --numa-aware=yes \
        --print-topology=yes \
        --mpi-global-args="--tag-output --allow-run-as-root  --mca oob_tcp_if_include $TCP_IF_INCLUDE --mca btl_tcp_if_include $TCP_IF_INCLUDE" \
        --mpi-local-args="-x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x CPATH -x IPUOF_VIPU_API_TIMEOUT=3600 -x POPLAR_LOG_LEVEL=WARN -x POPLAR_SDK_ENABLED -x POPLAR_ENGINE_OPTIONS" \
        --vipu-server-timeout=3600 \
python train_gpt2.py \
    --model gpt2-large \
    --max-len 512 \
    --optimizer AdamW \
    --learning-rate 0.00015 \
    --lr-schedule cosine \
    --lr-warmup 0.01 \
    --layers-per-ipu 0 5 5 5 5 5 5 6 \
    --matmul-proportion 0.40 0.12 0.15 0.15 0.15 0.15 0.15 0.10 \
    --ipus-per-replica 8 \
    --epochs 5 \
    --gradient-accumulation 1024 \
    --batches-per-step 8 \
    --batch-size 1 \
    --enable-sequence-serialized False \
    --embedding-serialization-factor 8 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --train-path 'generated' \
    --replicated-tensor-sharding True \
    --compile-only False