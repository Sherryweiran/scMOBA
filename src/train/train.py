import os
import sys
import shutil
import torch
import torch.nn.functional as F
from typing import Optional, List
import transformers
from transformers import Trainer, AutoTokenizer
from dataclasses import dataclass, field
from src.dataset.muti_dataset import UniDatasets
from src.model.mllm_gene import scLaGene
import numpy as np

from safetensors.torch import load_model, load_file
from peft import LoraConfig, get_peft_model

#change to your root data path
root = "./"
sys.path.append(root)
@dataclass
class AllArguments:
    # data related path
    # Users should place all datasets under this directory
    data_root: str = field(default="data", metadata={"help": "Root directory containing all datasets."})

    token_to_gene_dict_mouse_path: str = field(default="dataset/token2gene_mouse.pkl", metadata={"help": "Mouse gene symbol dictionary file."})
    token_to_gene_dict_human_path: str = field(default="dataset/token2gene_human.pkl", metadata={"help": "Human gene symbol dictionary file."})
    token_to_gene_dict_macaca_path: str = field(default="dataset/token2gene_macaca.pkl", metadata={"help": "Macaca gene symbol dictionary file."})
    token_to_gene_dict_macaque_path: str = field(default="dataset/token2gene_macaca.pkl", metadata={"help": "Macaque gene symbol dictionary file."})
    token_to_gene_dict_marmoset_path: str = field(default="dataset/token2gene_marmoset.pkl", metadata={"help": "Marmoset gene symbol dictionary file."})

    celltype_data_train_path: str = field(default="celltype/train", metadata={"help": "Training dataset for cell type classification."})
    celltype_data_eval_path: str = field(default="celltype/val", metadata={"help": "Validation dataset for cell type classification."})
    celltype_data_test_path: str = field(default="celltype/test", metadata={"help": "Testing dataset for cell type classification."})

    subclass_data_train_path: str = field(default="subclass/train", metadata={"help": "Training dataset for cell subclass classification."})
    subclass_data_eval_path: str = field(default="subclass/val", metadata={"help": "Validation dataset for cell subclass classification."})
    subclass_data_test_path: str = field(default="subclass/test", metadata={"help": "Testing dataset for cell subclass classification."})

    tissue_data_train_path: str = field(default="tissue/train", metadata={"help": "Training dataset for tissue classification."})
    tissue_data_eval_path: str = field(default="tissue/val", metadata={"help": "Validation dataset for tissue classification."})
    tissue_data_test_path: str = field(default="tissue/test", metadata={"help": "Testing dataset for tissue classification."})

    age_data_train_path: str = field(default="age/train", metadata={"help": "Training dataset for age prediction task."})
    age_data_eval_path: str = field(default="age/val", metadata={"help": "Validation dataset for age prediction task."})
    age_data_test_path: str = field(default="age/test", metadata={"help": "Testing dataset for age prediction task."})

    age_group_data_train_path: str = field(default="age_group/train", metadata={"help": "Training dataset for developmental stage prediction."})
    age_group_data_eval_path: str = field(default="age_group/val", metadata={"help": "Validation dataset for developmental stage prediction."})
    age_group_data_test_path: str = field(default="age_group/test", metadata={"help": "Testing dataset for developmental stage prediction."})

    disease_data_train_path: str = field(default="disease/train", metadata={"help": "Training dataset for disease classification."})
    disease_data_eval_path: str = field(default="disease/val", metadata={"help": "Validation dataset for disease classification."})
    disease_data_test_path: str = field(default="disease/test", metadata={"help": "Testing dataset for disease classification."})

    spatial_data_train_path: str = field(default="spatial/train", metadata={"help": "Training dataset for spatial transcriptomics task."})
    spatial_data_eval_path: str = field(default="spatial/val", metadata={"help": "Validation dataset for spatial transcriptomics task."})
    spatial_data_test_path: str = field(default="spatial/test", metadata={"help": "Testing dataset for spatial transcriptomics task."})
    # BERT config
    # =========================
    # NOTE:
    # The pretrained Geneformer model is NOT included in this repository.Please download it manually and place it under:
    # checkpoints/geneformer/gf-6L-30M-i2048
    bert_path: str = field(default="checkpoints/geneformer/gf-6L-30M-i2048", metadata={"help": "Path to pre-trained BERT model."})
    bert_frozen: bool = field(default=False,metadata={"help": "Whether to freeze BERT encoder."})
    bert_hidden_size: int = field(default=256, metadata={"help": "Hidden size of BERT model."})

    #  Projector 
    embedding_dim: int = field(default=2048, metadata={"help": "Embedding dimension for MLP projector."})
    llm_hidden_size: int = field(default=2048, metadata={"help": "Hidden size for LLM."})
    GCN_embedding_dim: int = field(default=2048, metadata={"help": "embedding size for GCN."})
    dropout: float = field(default=0.1, metadata={"help": "dropout for GCN."})
    num_querys:int = field(default=256, metadata={"help": "number of learnable query tokens."})
    encoder_num_heads:int = field(default=4, metadata={"help": "number of encoder headers"})
    training: bool = field(default=True,metadata={"help": "Whether to train GCN."})

    # =========================
    # LLM configuration
    # =========================
    # NOTE:
    # The pretrained LLM model is NOT included in this repository.# Please download the model (e.g., Llama-3.2-1B) and place it under:
    # checkpoints/llm/Llama-3.2-1B
    llm_path: str = field(default="checkpoints/llm/Llama-3.2-1B", metadata={"help": "Path to pre-trained LLM model."})
    tokenizer_path: str = field(default="checkpoints/llm/Llama-3.2-1B", metadata={"help": "Path to the tokenizer."})
    vocab_size: int = field(default=128000, metadata={"help": "Vocabulary size for the LLM model."})
    llm_frozen: bool = field(default=False, metadata={"help": "Whether to freeze LLM during training."})
    max_seq: int = field(default=512)


    # lora
    lora_enable: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_dropout: float = field(default=0.1)
    lora_bias: str = field(default="none")
    lora_task_type: str = field(default="CAUSAL_LM")

    # 
    pretrained_model_before_lora: str = field(default="", metadata={"help": "Path to pretrained model before applying LoRA."})
    pretrained_model_after_lora: str = field(default="", metadata={"help": "Path to pretrained model after applying LoRA."})


def compute_metrics(eval_preds):
    # predictions=all_preds, label_ids=all_labels, inputs=all_inputs
    # N*seq
    result = dict()
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds==filtered_labels) / len(filtered_labels)
    result["accuracy"] = acc_score

    return result

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids



@dataclass
class DataCollator:
    def __init__(self, args):
        self.args = args
    def __call__(self, batch: list) -> dict:
     
        images, input_ids, labels, attention_masks = tuple(
        [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask'))
        
        max_seq_len = max(tensor.shape[-1] for tensor in images)
        padded_images = []
        original_composition = []
        for tensor in images:
    
            pad_size = max_seq_len - tensor.shape[-1]
       
            padded_tensor = F.pad(tensor, (0, pad_size), "constant", 0)
            if len(padded_tensor.shape)==1:
                padded_tensor = padded_tensor.unsqueeze(0)
            size = padded_tensor.shape[0]
            padded_images.append(padded_tensor)
            original_composition.append(size)
        images = torch.cat(padded_images, dim=0)
        images = images.to(torch.int64)  
        input_ids = torch.stack(input_ids, dim=0)      
        labels = torch.stack(labels, dim=0)             
        attention_masks = torch.stack(attention_masks, dim=0) 
       # original_composition = torch.tensor(
       #     original_composition,
        #    dtype=torch.long,
        #    device=images.device,
       # )
       # print("data collator image shape:",images.shape)
        #print("data collator composition:",sum(original_composition))
        return_dict = dict(
            images=images,
            input_ids=input_ids,
            labels=labels,
            attention_masks=attention_masks,
            original_composition = original_composition
            )

        return return_dict
def main():
    parser = transformers.HfArgumentParser(AllArguments)
    args = parser.parse_args_into_dataclasses()[0]

    training_args = transformers.TrainingArguments(
        do_train=True,
        do_eval=False, 
        # do_eval=True,
        per_device_train_batch_size=36, #24,
        per_device_eval_batch_size=1,

        num_train_epochs=2,#15, 15, pretrain, finetune
        #evaluation_strategy="steps",
        #eval_accumulation_steps=1,
        #eval_steps=0.1,

        learning_rate=1e-5, #1e-4, 2e-5
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        # weight_decay=0,
        optim="adamw_torch",
        bf16=True,

        dataloader_pin_memory=True,
        dataloader_num_workers=8,
        remove_unused_columns=False,

        report_to="tensorboard",
        output_dir=root + "outputs",
        logging_steps=0.001,
        save_steps=2000,
        save_total_limit=2,
        save_safetensors=False
    )

    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, model_max_length=args.max_seq, padding_side="right", use_fast=False)

    # Define and add special tokens
    special_image_tokens = ["<im_patch>"]
    special_token = {"additional_special_tokens": special_image_tokens}
    tokenizer.add_special_tokens(
        special_token
    )

    # Set special token IDs for padding, beginning of sequence, and end of sequence
    tokenizer.pad_token_id = 128004
    tokenizer.bos_token_id = 128000
    tokenizer.eos_token_id = 128001

    # Convert special tokens to token IDs and set related arguments
    
    args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    args.vocab_size = len(tokenizer)
    print("len tokenizer: ", len(tokenizer))

    tokenizer.save_pretrained(os.path.join(training_args.output_dir, "tokenizer"))

    print("="*20 + " Dataset preparation " + "="*20)
    train_dataset = UniDatasets(args, tokenizer, mode='train')
    print(">>> Finished train_dataset", flush=True)
    #eval_dataset = UniDatasets(args, tokenizer, mode='validation')
    data_collator = DataCollator(args)
    print(">>> Finished data_collator", flush=True)


    print("="*20 + " Model preparation " + "="*20, flush=True)
    model = scLaGene(args)

    if args.pretrained_model_before_lora:
        #load_model(model, args.pretrained_mllm_model, strict=False)
        model.load_state_dict(torch.load(args.pretrained_model_before_lora))
        print("Load pretrained MLLM without LoRA")

    if args.lora_enable:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type=args.lora_task_type
        )
        model.llm = get_peft_model(model.llm, lora_config)
        model.llm.print_trainable_parameters()
        print("Add LoRA on LLM")

    if args.pretrained_model_after_lora:
        model.load_state_dict(torch.load(args.pretrained_model_after_lora), strict=False)
        print("Load pretrained MLLM with LoRA")

    #model.config.to_json_file(os.path.join(training_args.output_dir, "config.json"))


    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_dataset,
                      #eval_dataset=eval_dataset,
                      compute_metrics=compute_metrics,
                      preprocess_logits_for_metrics=preprocess_logits_for_metrics
                      )

    #save train file and configs
    source_path = root+"/src/train/train.py"
    os.makedirs(training_args.output_dir, exist_ok=True)
    destination_path = os.path.join(training_args.output_dir, "train.py")
    shutil.copyfile(source_path, destination_path)

    trainer.train()
    trainer.save_state()
      
if __name__ == "__main__":
    main()
