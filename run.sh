#!/bin/bash

# export OPENAI_API_KEY="sk-yMe77ufJzUA5w3WqwNRZtlipG3jPQQrDptCsi01VXhDWDDoe"

# python meta.py \
#     --dataset citeseer \
#     --ptb_rate 0.05 \
#     --use_text_attack \
#     --llm_type gpt \
#     --openai_api_key $OPENAI_API_KEY \
#     --api_base_url  https://yibuapi.com/v1 \


export OLLAMA_MODEL="llama3.2:3b-instruct-fp16"

python meta.py \
    --dataset citeseer \
    --ptb_rate 0.05 \
    --use_text_attack \
    --llm_type gpt \
    --openai_api_key "ollama" \
    --api_base_url "http://localhost:11434/v1" \
    --text_retries 0 \
