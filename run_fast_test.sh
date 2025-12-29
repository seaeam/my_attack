#!/bin/bash

# 快速测试版本 - 只攻击20个节点
# 预计时间: 约40-100秒

# 请先在终端设置环境变量：
# export OPENAI_API_KEY="your-deepseek-api-key"

# 检查API密钥是否设置
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 错误: OPENAI_API_KEY 环境变量未设置"
    echo "请先运行: export OPENAI_API_KEY='your-deepseek-api-key'"
    exit 1
fi

echo "🚀 Running FAST TEST mode (20 nodes only)"
echo "⏱️  Estimated time: 40-100 seconds"
echo ""

python meta.py \
    --dataset citeseer \
    --ptb_rate 0.05 \
    --use_text_attack \
    --llm_type deepseek \
    --openai_api_key $OPENAI_API_KEY \
    --api_base_url https://api.deepseek.com \
    --text_attack_nodes 20
