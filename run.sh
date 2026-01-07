#!/bin/bash

export OPENAI_API_KEY="sk-yMe77ufJzUA5w3WqwNRZtlipG3jPQQrDptCsi01VXhDWDDoe"

python meta.py \
    --dataset citeseer \
    --ptb_rate 0.05 \
    --use_text_attack \
    --llm_type gpt \
    --openai_api_key $OPENAI_API_KEY \
    --api_base_url  https://yibuapi.com/v1 \
    
