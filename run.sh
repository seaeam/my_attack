#!/bin/bash

export OLLAMA_MODEL="llama3.2:1b-instruct-fp16"

python meta.py \
    --dataset citeseer \
    --model Meta-Both \
    --ptb_rate 0.20 \
    --level 4 \
    --step 1 \
    --miter 50 \
    --lr 0.05 \
    --global_important_ratio 0.35 \
    --global_ppr_alpha 0.10 \
    --global_ppr_iters 100 \
    --global_seed_strategy degree \
    --use_text_attack \
    --llm_type gpt \
    --openai_api_key "ollama" \
    --api_base_url "http://localhost:11434/v1" \
    --text_retries 1 \
    --text_budget_per_node 25 \
    --text_topk_ratio 0.08 \
    --text_ppr_alpha 0.25 \
    --text_ppr_iters 30 \
    --text_min_cluster_size 2 \
    --text_max_cluster_size 6 \
