#!/bin/bash

export OPENAI_API_KEY="sk-7f399d652cbb444490c83733dcf3687c"

python meta.py \
    --dataset citeseer \
    --ptb_rate 0.05 \
    --use_text_attack \
    --llm_type deepseek \
    --openai_api_key $OPENAI_API_KEY \
    --api_base_url https://api.deepseek.com
