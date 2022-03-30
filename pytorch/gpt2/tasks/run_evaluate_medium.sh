#!/bin/bash

echo "Evaluate using $1 dataset"

if [ $1 == "wiki" ]
then
  python tasks/evaluate_wiki.py \
        --valid-data data/wikitext-103/wiki.test.tokens \
        --pretrained-checkpoint checkpoints/gpt2_medium_50264_1024_dynamic_2/step_48501 \
        --fp16 \
        --seq-length 1024 \
        --tokenizer-type 1 \
        --batches-per-step 1 \
        --layers-per-ipu 7 8 8 1 \
        --matmul-proportion 0.4 0.4 0.4 0.2 \
        --executable-cache-dir /localdata/cn-customer-engineering/chaon/tmp
elif [ $1 == "lmbd" ]
then
  python tasks/evaluate_lambada.py \
        --valid-data data/lambada_test.jsonl \
        --pretrained-checkpoint checkpoints/gpt2_medium_50264_1024_dynamic_2/step_48501 \
        --fp16 \
        --seq-length 1024 \
        --tokenizer-type 1 \
        --batches-per-step 1 \
        --layers-per-ipu 7 8 8 1 \
        --matmul-proportion 0.4 0.4 0.4 0.2 \
        --strict-lambada false \
        --executable-cache-dir /localdata/cn-customer-engineering/chaon/tmp
else
  echo "Dataset should be 'wiki' or 'lmbd'"
fi
