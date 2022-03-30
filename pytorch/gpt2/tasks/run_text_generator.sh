#!/bin/bash
# run gpt2-small on single IPU
python text_generate_gpt2.py \
      --model-name-or-path gpt2 \
      --fp16 true \
      --single-ipu true \
      --poptorch-loop true \
      --output-len 256

# run gpt2-medium on 4 IPUs
# python text_generate_gpt2.py \
#       --model-name-or-path gpt2-medium \
#       --fp16 true \
#       --single-ipu false \
#       --poptorch-loop false \
#       --layers-per-ipu 1 7 8 8 \
#       --matmul-proportion 0.2 0.2 0.2 0.2 \
#       --output-len 256