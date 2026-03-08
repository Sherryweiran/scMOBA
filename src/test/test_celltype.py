import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from src.dataset.muti_dataset import CellTypeDataset

from src.model.mllm_gene import scLaGene
from src.train.train import AllArguments
from peft import LoraConfig, get_peft_model
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk

# ===============================
# Path to the dataset used to build the cell type dictionary
# Users should replace this with their own dataset path if needed.
# The dataset must contain a column named "Class".
# Example:
# dataset_path = "data/mouse_RNA_class"
# Load cell type dictionary
dataset_path = "Class_dataset"
dataset = load_from_disk(dataset_path)
# Get unique values from the 'Class' column
loaded_cell_types = dataset.unique('Class')
print(f"Loaded cell types: {loaded_cell_types}")
print(f"Number of unique cell types: {len(loaded_cell_types)}")
# Convert results to list
loaded_cell_types_list = list(loaded_cell_types)
print(loaded_cell_types_list)

# Function to extract and normalize cell type
def extract_and_standardize_cell_type(text, loaded_cell_types):
    for standard_cell_type in loaded_cell_types:
        if standard_cell_type.lower() in text.lower():
            return standard_cell_type
    return None
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(42)

# Postprocess predictions and labels
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def main():
    parser = transformers.HfArgumentParser(AllArguments)
    args = parser.parse_args_into_dataclasses()[0]
        # ==========================================
    # Path to tokenizer
    #
    # Users should place the tokenizer folder here.
    # Example structure:
    #
    # checkpoints/
    #    tokenizer/
    #        tokenizer_config.json
    #        special_tokens_map.json
    #        tokenizer.json
    #
    # If you trained your own tokenizer, update this path.
    ## Load tokenizer
    tokenizer_path ="checkpoints/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,model_max_length=args.max_seq, padding_side="right")
    # Configure model and LoRA
    args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    args.vocab_size = len(tokenizer)
    print("len tokenizer: ", len(tokenizer))
    model = scLaGene(args)

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
     # ==========================================
    # Path to trained model checkpoint
    # Replace this with your trained model path.
    #
    # Example:
    # checkpoints/model.bin or checkpoints/checkpoint-100000/pytorch_model.bin
    model_weights_path = "checkpoints/model.bin"
    model.load_state_dict(torch.load(model_weights_path), strict=True)
    print("Load trained MLLM ")

    print("Setup Data")
    test_dataset = CellTypeDataset(args, tokenizer, mode='test')

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=32,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )  

    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
     # ==========================================
    # Path to save prediction results
    #
    # The output CSV will contain:
    # question, prediction, correctness, etc.
    # ==========================================
    test_res_path = "./class_result.csv"

    results = []

    for sample in tqdm(test_dataloader):
        question = sample["question"]
        question_type = sample["question_type"]
        answer = sample['answer']

        image = sample["image"].to(device)
        input_id = tokenizer(question, return_tensors="pt")['input_ids'].to(device)

        # Generate model output
        generation = model.generate(image, input_id)
        generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

        # Extract normalized cell type
        standard_cell_type = extract_and_standardize_cell_type(answer[0], loaded_cell_types)
        generated_cell_type = extract_and_standardize_cell_type(generated_texts[0], loaded_cell_types)

        # Compare predicted cell type with ground truth
        correct = 0
        if generated_cell_type and standard_cell_type:
            if generated_cell_type.lower() == standard_cell_type.lower():
                correct = 1

        results.append([question_type, question[0], answer[0], generated_texts[0], correct, standard_cell_type, generated_cell_type])

    # Save prediction results to CSV
    with open(test_res_path, mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Question Type", "Question", "Answer", "Pred", "Correct", "Standard Cell Type", "Generated Cell Type"])
        writer.writerows(results)

if __name__ == "__main__":
    main()
