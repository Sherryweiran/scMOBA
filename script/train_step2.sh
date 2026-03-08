#!/bin/bash
# run "accelerate config" first!

accelerate launch src/train/train.py \
    --bert_path checkpoints/geneformer/gf-6L-30M-i2048 \
    --llm_path checkpoints/llm/Llama-3.2-1B \
    --tokenizer_path checkpoints/llm/Llama-3.2-1B \
    --bert_frozen False \
    --llm_frozen False \
    --lora_enable True \
    --pretrained_model_before_lora outputs/step1_alignment/pytorch_model.bin \
    --training True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 36 \
    --bf16 True \
    --output_dir outputs/step2_instruction \
    --report_to tensorboard