#!/bin/bash
# run "accelerate config" first!
accelerate launch src/train/train.py \
    --bert_path checkpoints/geneformer/gf-6L-30M-i2048 \
    --llm_path checkpoints/llm/Llama-3.2-1B \
    --tokenizer_path checkpoints/llm/Llama-3.2-1B \
    --bert_frozen True \
    --llm_frozen True \
    --lora_enable False \
    --training True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 36 \
    --bf16 True \
    --output_dir outputs/step1_alignment \
    --report_to tensorboard