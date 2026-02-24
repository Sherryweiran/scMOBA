import sys
import random
import os
from datasets.arrow_dataset import sample
import numpy as np
from dataclasses import dataclass, field
from datasets import load_from_disk
import tqdm
import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence

import pickle

import json
import pandas as pd



class CellTypeDataset(Dataset):
    def __init__(self, args, tokenizer, description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.description = description
        self.num_querys = 256
       
        #self.box_tokens = ["<bx_start>", "<bx_end>"]

        if mode == "train":
            self.data_list = load_from_disk(args.celltype_data_train_path)
        elif mode == "validation":
            self.data_list = load_from_disk(args.celltype_data_eval_path)
        elif mode == "test":
            self.data_list = load_from_disk(args.celltype_data_test_path)

        #with open(self.args.term_dict_path, 'r') as file:
        #    self.term_dict = json.load(file) #term dictionanry

        self.question_list = [
            "Can you predict the cell type based on this gene expression data? Please provide your answer.",
            "Based on the gene expression data, what cell type do you think is present?",
            "Can you identify the cell type associated with the given gene expression profile?",
            "What cell type corresponds to this gene expression pattern? Please state your prediction.",
            "Based on these gene expression markers, what cell type is most likely present?",
            "Can you deduce the cell type from the gene expression data? ",
            "What cell type is most likely to be present given this gene expression profile?",
            "Using the provided gene expression data, can you infer the cell type? Please state your conclusion.",
            "Can you identify the cell type associated with the expression of these genes?",
            "Based on the gene expression data, which cell type might be represented in this sample?"
        ]

        self.answer_list = [
            "The predicted cell type is {}.",
            "Based on the data, the cell type is {}.",
            "The cell type is {}.",
            "Based on the gene expression, the identified cell type is {}.",
            "The cell type is {}.",
            "Here is the predicted cell type: {}.",
            "Based on this gene expression profile, it's most likely the cell type {}.",
            "The inferred cell type is {}.",
            "The identified cell type is {}.",
            "The cell type predicted from the gene expression is {}."]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_array = data['input_ids']
            image_array = torch.tensor(image_array)

            cell_type = data['Class'] #

            #sample_id = data['cell_label']
            #batch_label = data['batch']
            species = data['species']
            seq_method = data['seq_method']
            #slice = data['slice']
    

            self.image_tokens = "<im_patch>" * self.num_querys
            # 添加调试信息
            
            #print(self.description)
            #print(self.question_list)

            #try:
            question_temple = random.choice(self.question_list)
            #print("this is question temple",question_temple)
            question = self.image_tokens + ' ' + question_temple
            #print("this is question",question)
            answer = random.choice(self.answer_list).format(cell_type)

            text_tensor = self.tokenizer(
                question + ' ' + answer, max_length=self.args.max_seq, truncation=True, padding="max_length",
                return_tensors="pt"
            )
            input_id = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]

            valid_len = torch.sum(attention_mask)
            if valid_len < len(input_id):
                input_id[valid_len] = self.tokenizer.eos_token_id

            question_tensor = self.tokenizer(
                question, max_length=self.args.max_seq, truncation=True, padding="max_length", return_tensors="pt"
            )
            question_len = torch.sum(question_tensor["attention_mask"][0])

            label = input_id.clone()
            label[label == self.tokenizer.pad_token_id] = -100
            label[:question_len] = -100
            #print(self.description)

            ret = {
                'image': image_array,
                'input_id': input_id,
                'label': label,
                'attention_mask': attention_mask,
                'question': question,
                'answer': answer,
                'question_type': "Celltype",
                'cell_type': cell_type,
                'species':species,
                'seq_method':seq_method
                #"batch_label": batch_label
                #'sample_id': sample_id
                #'slice':slice
                
            }
            return ret

            #except Exception as e:
            #   print(f"Error in __getitem__ at index {idx}: {e}")
            #   idx = random.randint(0, len(self.data_list) - 1)


class SubclassDataset(Dataset):
    def __init__(self, args, tokenizer, description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.description = description
        self.num_querys = 256
       
        #self.box_tokens = ["<bx_start>", "<bx_end>"]

        if mode == "train":
            self.data_list = load_from_disk(args.subclass_data_train_path)
        elif mode == "validation":
            self.data_list = load_from_disk(args.subclass_data_eval_path)
        elif mode == "test":
            self.data_list = load_from_disk(args.subclass_data_test_path)

        #with open(self.args.term_dict_path, 'r') as file:
        #    self.term_dict = json.load(file) #term dictionanry

        self.question_list = [
            "Can you predict the cell subtype based on this gene expression data? Please provide your answer.",
            "Based on the gene expression data, what cell subtype do you think is present?",
            "Can you identify the cell subclass associated with the given gene expression profile?",
            "What cell subtype corresponds to this gene expression pattern? Please state your prediction.",
            "Based on these gene expression markers, what cell subclass is most likely present?",
            "Can you deduce the cell subtype from the gene expression data?",
            "What cell subtype is most likely to be present given this gene expression profile?",
            "Using the provided gene expression data, can you infer the cell subclass? Please state your conclusion.",
            "Can you identify the cell subtype associated with the expression of these genes?",
            "Based on the gene expression data, which cell subclass might be represented in this sample?",
            "Can you determine the more specific cell type (subtype) from this expression pattern?",
            "Based on these molecular markers, can you identify the specific cell subtype?",
            "Can you predict the cellular subpopulation represented by this expression profile?"]

        self.answer_list = [
            "The predicted cell subtype is {}.",
            "Based on the data, the cell subclass is {}.",
            "The cell subtype is {}.",
            "Based on the gene expression, the identified cell subclass is {}.",
            "The cellular subpopulation is {}.",
            "Here is the predicted cell subtype: {}.",
            "Based on this gene expression profile, it's most likely the cell subtype {}.",
            "The inferred cell subclass is {}.",
            "The cell subtype predicted from the gene expression is {}.",
            "This expression pattern suggests the {} cell subpopulation.",
            "The molecular signature corresponds to {} subtype.",
            "This appears to be the {} subpopulation of cells.",
            "The detailed cell type prediction is {}."]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_array = data['input_ids']
            image_array = torch.tensor(image_array)

            #Subclass = data['Supertype'] # 
            Subclass = data['Subclass_ori'] # 
            #Subclass = data['clusterName'] 
            #Subclass = data['BICCN_cluster_label'] 
            #soma_id = data['soma_joinid']
            #sample_id = data['name']
            batch_labels = data['domain_labels']
    

            self.image_tokens = "<im_patch>" * self.num_querys
            # 添加调试信息
            
            #print(self.description)
            #print(self.question_list)

            #try:
            question_temple = random.choice(self.question_list)
            #print("this is question temple",question_temple)
            question = self.image_tokens + ' ' + question_temple
            #print("this is question",question)
            answer = random.choice(self.answer_list).format(Subclass)

            text_tensor = self.tokenizer(
                question + ' ' + answer, max_length=self.args.max_seq, truncation=True, padding="max_length",
                return_tensors="pt"
            )
            input_id = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]

            valid_len = torch.sum(attention_mask)
            if valid_len < len(input_id):
                input_id[valid_len] = self.tokenizer.eos_token_id

            question_tensor = self.tokenizer(
                question, max_length=self.args.max_seq, truncation=True, padding="max_length", return_tensors="pt"
            )
            question_len = torch.sum(question_tensor["attention_mask"][0])

            label = input_id.clone()
            label[label == self.tokenizer.pad_token_id] = -100
            label[:question_len] = -100
            #print(self.description)

            ret = {
                'image': image_array,
                'input_id': input_id,
                'label': label,
                'attention_mask': attention_mask,
                'question': question,
                'answer': answer,
                'question_type': "Subclass",
                'Subclass': Subclass
                #"sample_id": sample_id
                
            }
            return ret

            #except Exception as e:
            #   print(f"Error in __getitem__ at index {idx}: {e}")
            #   idx = random.randint(0, len(self.data_list) - 1)



class TissueTypeDataset(Dataset):
    def __init__(self, args, tokenizer, description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.description = description
        self.num_querys = 256  # Number of tokens representing tissue-related information
        
        if mode == "train":
            self.data_list = load_from_disk(args.vqa_data_train_path)
        elif mode == "validation":
            self.data_list = load_from_disk(args.vqa_data_val_path)
        elif mode == "test":
            self.data_list = load_from_disk(args.vqa_data_test_path)
        
        print(len(self.data_list))
        self.data_list = self.data_list.filter(lambda x: x['tissue'] != None,num_proc=50)
        print(len(self.data_list))

        # Define question templates for tissue classification
        self.question_list = [
            "Predict the tissue type based on this gene expression data.",
            "Based on the gene expression data, what tissue type is present?",
            "Identify the tissue type associated with the given gene expression profile.",
            "What tissue type corresponds to this gene expression pattern? Please state your prediction.",
            "Based on these gene expression markers, what tissue type is most likely present?",
            "Deduce the tissue type from the gene expression data.",
            "What tissue type is most likely to be present given this gene expression profile?",
            "Using the provided gene expression data, infer the tissue type. Please state your conclusion.",
            "Identify the tissue type associated with the expression of these genes.",
            "Based on the gene expression data, which tissue type might be represented in this sample?",
            "What tissue type is most strongly associated with the following gene expression data?",
            "From this gene expression profile, determine the tissue type involved.",
            "Classify the tissue type based on the gene expression data provided.",
            "Given this gene expression data, predict which tissue type is present.",
            "Based on the markers present, determine the tissue type indicated by the gene expression.",
            "Determine the tissue type corresponding to these gene expression levels.",
            "What is the most probable tissue type associated with the gene expression data?",
            "Infer the tissue type from this set of gene expression data.",
            "Identify the tissue type suggested by this gene expression pattern.",
            "Match this gene expression data to the correct tissue type."
            
        ]

        # Updated answer list without "Sure" and "Of course"
        self.answer_list = [
            "The sample is from {}.",
            "The predicted tissue type is {}.",
            "Based on the data, the tissue type is {}.",
            "The tissue type is {}.",
            "Based on the gene expression, the identified tissue type is {}.",
            "Here is the predicted tissue type: {}.",
            "Based on this gene expression profile, it's most likely the tissue type {}.",
            "The inferred tissue type is {}.",
            "The identified tissue type is {}.",
            "The tissue type predicted from the gene expression is {}."
        ]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_array = data['input_ids']
            image_array = torch.tensor(image_array)

            tissue_type = data['tissue']

            # Placeholder token to represent tissue-related patches
            self.image_tokens = "<im_patch>" * self.num_querys


            # Randomly select a question and answer based on tissue type
            question_temple = random.choice(self.question_list)
            question = self.image_tokens + ' ' + question_temple
            answer = random.choice(self.answer_list).format(tissue_type)

            # Tokenize question and answer together
            text_tensor = self.tokenizer(
                question + ' ' + answer, max_length=self.args.max_seq, truncation=True, padding="max_length",
                return_tensors="pt"
            )
            input_id = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]

            # Handle the attention mask and input ID length
            valid_len = torch.sum(attention_mask)
            if valid_len < len(input_id):
                input_id[valid_len] = self.tokenizer.eos_token_id

            # Tokenize only the question for length calculation
            question_tensor = self.tokenizer(
                question, max_length=self.args.max_seq, truncation=True, padding="max_length", return_tensors="pt"
            )
            question_len = torch.sum(question_tensor["attention_mask"][0])

            # Create labels with proper masking
            label = input_id.clone()
            label[label == self.tokenizer.pad_token_id] = -100
            label[:question_len] = -100

            ret = {
                'image': image_array,
                'input_id': input_id,
                'label': label,
                'attention_mask': attention_mask,
                'question': question,
                'answer': answer,
                'question_type': "TissueType",  # Changed to TissueType for this dataset
            }
            return ret


class DevelopmentStageDataset(Dataset):
    def __init__(self, args, tokenizer, description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.description = description
        self.num_querys = 256  # Number of tokens representing the development-related information
        
        if mode == "train":
            self.data_list = load_from_disk(args.vqa_data_train_path)
        elif mode == "validation":
            self.data_list = load_from_disk(args.vqa_data_val_path)
        elif mode == "test":
            self.data_list = load_from_disk(args.vqa_data_test_path)
        
        print(len(self.data_list))
        self.data_list = self.data_list.filter(lambda x: x['development_stage'] != None,num_proc=50)
        print(len(self.data_list))


        # Define question templates for development stage prediction
        self.question_list = [
            "What is the most likely age of this sample based on its gene expression?",
            "Predict the age from this data.",
            "How old is this sample, given its gene expression profile?",
            "Estimate the age using these transcriptional signatures.",
            "This data suggests what biological age?",
            "Determine the age of the cell sample from its gene expression.",
            "Infer the chronological age based on this transcriptome.",
            "What age does this gene expression pattern indicate?",
            "Guess the age of the donor using these expression markers.",
            "From these gene levels, calculate the probable age."
        ]

        # Define answer templates (adjusted for development stage)
        self.answer_list = [
            "The most likely age of this sample is {} based on its gene expression profile.",
            "The predicted age is {}.",
            "Given its gene expression profile, this sample appears to be from a {} individual.",
            "The age estimate using these signatures is {}.",
            "This data suggests a biological age of {}.",
            "The determined age of the cell sample is {} based on gene expression analysis.",
            "The inferred chronological age from this transcriptome is {}.",
            "This gene expression pattern indicates an age of {}.",
            "The donor's age is estimated to be {} based on these expression markers.",
            "The calculated probable age from these gene expression levels is {}."
        ]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_array = data['input_ids']
            image_array = torch.tensor(image_array)

            development_stage = data['development_stage']

            # Placeholder token to represent development-related patches
            self.image_tokens = "<im_patch>" * self.num_querys

   
            # Randomly select a question and answer based on the development stage
            question_temple = random.choice(self.question_list)
            question = self.image_tokens + ' ' + question_temple
            answer = random.choice(self.answer_list).format(development_stage)

            # Tokenize question and answer together
            text_tensor = self.tokenizer(
                question + ' ' + answer, max_length=self.args.max_seq, truncation=True, padding="max_length",
                return_tensors="pt"
            )
            input_id = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]

            # Handle the attention mask and input ID length
            valid_len = torch.sum(attention_mask)
            if valid_len < len(input_id):
                input_id[valid_len] = self.tokenizer.eos_token_id

            # Tokenize only the question for length calculation
            question_tensor = self.tokenizer(
                question, max_length=self.args.max_seq, truncation=True, padding="max_length", return_tensors="pt"
            )
            question_len = torch.sum(question_tensor["attention_mask"][0])

            # Create labels with proper masking
            label = input_id.clone()
            label[label == self.tokenizer.pad_token_id] = -100
            label[:question_len] = -100

            ret = {
                'image': image_array,
                'input_id': input_id,
                'label': label,
                'attention_mask': attention_mask,
                'question': question,
                'answer': answer,
                'question_type': "DevelopmentStage",  # Changed to DevelopmentStage for this dataset
            }
            return ret

class AgeGroupDataset(Dataset):
    def __init__(self, args, tokenizer, description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.description = description
        self.num_querys = 256  # Number of tokens representing the development-related information
        
        if mode == "train":
            self.data_list = load_from_disk(args.vqa_data_train_path)
        elif mode == "validation":
            self.data_list = load_from_disk(args.vqa_data_val_path)
        elif mode == "test":
            self.data_list = load_from_disk(args.vqa_data_test_path)
        
        print(len(self.data_list))
        self.data_list = self.data_list.filter(lambda x: x['age_group'] != None,num_proc=50)
        print(len(self.data_list))


        # Define question templates for development stage prediction
        self.question_list = [
            "What is the most likely developmental stage of this sample based on its gene expression?",
            "Predict the age from this data.",
            "How old is this sample, given its gene expression profile?",
            "Estimate the age using these transcriptional signatures.",
            "This data suggests what biological age?",
            "Determine the age of the cell sample from its gene expression.",
            "Infer the chronological age based on this transcriptome.",
            "What age does this gene expression pattern indicate?",
            "Guess the age of the donor using these expression markers.",
            "From these gene levels, calculate the probable age."
        ]

        # Define answer templates (adjusted for development stage)
        self.answer_list = [
            "The most likely development stage of this sample is {} based on its gene expression profile.",
            "The predicted development stage is {}.",
            "Given its gene expression profile, this sample appears to be from a {} individual.",
            "The development stage estimate using these signatures is {}.",
            "This data suggests a biological development stage of {}.",
            "The determined development stage of the cell sample is {} based on gene expression analysis.",
            "The inferred chronological development stage from this data is {}.",
            "This gene expression pattern indicates an development stage of {}.",
            "The donor's development stage is estimated to be {} based on these expression markers.",
            "The calculated probable development stage from these gene expression levels is {}."
        ]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_array = data['input_ids']
            image_array = torch.tensor(image_array)

            development_stage = data['development_stage']

            # Placeholder token to represent development-related patches
            self.image_tokens = "<im_patch>" * self.num_querys

   
            # Randomly select a question and answer based on the development stage
            question_temple = random.choice(self.question_list)
            question = self.image_tokens + ' ' + question_temple
            answer = random.choice(self.answer_list).format(development_stage)

            # Tokenize question and answer together
            text_tensor = self.tokenizer(
                question + ' ' + answer, max_length=self.args.max_seq, truncation=True, padding="max_length",
                return_tensors="pt"
            )
            input_id = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]

            # Handle the attention mask and input ID length
            valid_len = torch.sum(attention_mask)
            if valid_len < len(input_id):
                input_id[valid_len] = self.tokenizer.eos_token_id

            # Tokenize only the question for length calculation
            question_tensor = self.tokenizer(
                question, max_length=self.args.max_seq, truncation=True, padding="max_length", return_tensors="pt"
            )
            question_len = torch.sum(question_tensor["attention_mask"][0])

            # Create labels with proper masking
            label = input_id.clone()
            label[label == self.tokenizer.pad_token_id] = -100
            label[:question_len] = -100

            ret = {
                'image': image_array,
                'input_id': input_id,
                'label': label,
                'attention_mask': attention_mask,
                'question': question,
                'answer': answer,
                'question_type': "AgeGroup",  # Changed to DevelopmentStage for this dataset
            }
            return ret


class DiseaseDataset(Dataset):
    def __init__(self, args, tokenizer, description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.description = description
        self.num_querys = 256  # Number of tokens representing the disease-related information
        
        if mode == "train":
            self.data_list = load_from_disk(args.vqa_data_train_path)
        elif mode == "validation":
            self.data_list = load_from_disk(args.vqa_data_val_path)
        elif mode == "test":
            self.data_list = load_from_disk(args.vqa_data_test_path)
        
        print(len(self.data_list))
        self.data_list = self.data_list.filter(lambda x: x['disease'] != None,num_proc=50)
        print(len(self.data_list))


        # Define question templates for disease prediction
        self.question_list = [
            "Diagnose the disease from this gene expression profile.",
            "What disease does this expression pattern indicate?",
            "Predict the disease using this transcriptomic data.",
            "Identify the pathological condition from these gene markers.",

            "Analyze this expression data to determine the disease.",
            "Based on these biomarkers, what is the likely diagnosis?",
            "Which disease matches this transcriptional signature?",
            "Interpret this gene expression profile to predict the disease.",
            "What pathological state does this data represent?",
            "Determine the disease from this expression pattern.",
            "Classify the disease subtype using this data.",
            "Which disorder is suggested by these dysregulated genes?",
            "Predict the disease based on this gene expression data.",
            "Based on the gene expression data, what disease is present?",
            "Identify the disease associated with the given gene expression profile.",
            "What disease corresponds to this gene expression pattern? Please state your prediction.",
            "Based on these gene expression markers, what disease is most likely present?",
            "Deduce the disease from the gene expression data.",
            "What disease is most likely to be present given this gene expression profile?",
            "Identify the disease associated with the expression of these genes.",
            "Based on the gene expression data, which disease might be represented in this sample?",
            "What disease is most strongly associated with the following gene expression data?",
            "From this gene expression profile, determine the disease involved.",
            "Classify the disease based on the gene expression data provided.",
            "Given this gene expression data, predict which disease is present.",
            "Based on the markers present, determine the disease indicated by the gene expression.",
            "Determine the disease corresponding to these gene expression levels.",
            "What is the most probable disease associated with the gene expression data?",
            "Infer the disease from this set of gene expression data.",
            "Identify the disease suggested by this gene expression pattern.",
        ]

        # Define answer templates (adjusted for disease prediction)
        self.answer_list = [
            "Predicted disease: {}.",
            "This sample indicates {}.",
            "Most likely disease: {}.",
            "It is associated with {}.", 
            "The diagnostic conclusion from this profile is {}.",
            "This expression pattern indicates {}.",
            "The data predicts {}.",
            "These gene markers identify {} as the pathological condition.",
            "Analysis of this expression data reveals {}.",
            "The biomarker profile suggests a diagnosis of {}.",
            "This transcriptional signature best matches {}.",
            "Profile interpretation predicts {} with high confidence.",
            "The predicted disease is {}.",
            "Based on the data, the disease is {}.",
            "The disease is {}.",
            "Based on the gene expression, the identified disease is {}.",
            "Here is the predicted disease: {}.",
            "Based on this gene expression profile, it's most likely the disease {}.",
            "The inferred disease is {}.",
            "The identified disease is {}.",
            "The disease predicted from the gene expression is {}."
        ]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_array = data['input_ids']
            image_array = torch.tensor(image_array)

            disease = data['disease']

            # Placeholder token to represent disease-related patches
            self.image_tokens = "<im_patch>" * self.num_querys

        
            # Randomly select a question and answer based on the disease
            question_temple = random.choice(self.question_list)
            question = self.image_tokens + ' ' + question_temple
            answer = random.choice(self.answer_list).format(disease)

            # Tokenize question and answer together
            text_tensor = self.tokenizer(
                question + ' ' + answer, max_length=self.args.max_seq, truncation=True, padding="max_length",
                return_tensors="pt"
            )
            input_id = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]

            # Handle the attention mask and input ID length
            valid_len = torch.sum(attention_mask)
            if valid_len < len(input_id):
                input_id[valid_len] = self.tokenizer.eos_token_id

            # Tokenize only the question for length calculation
            question_tensor = self.tokenizer(
                question, max_length=self.args.max_seq, truncation=True, padding="max_length", return_tensors="pt"
            )
            question_len = torch.sum(question_tensor["attention_mask"][0])

            # Create labels with proper masking
            label = input_id.clone()
            label[label == self.tokenizer.pad_token_id] = -100
            label[:question_len] = -100

            ret = {
                'image': image_array,
                'input_id': input_id,
                'label': label,
                'attention_mask': attention_mask,
                'question': question,
                'answer': answer,
                'question_type': "Disease",  # Changed to Disease for this dataset
            }
            return ret

class MaskedGeneDataset(Dataset):
    def __init__(self, args, tokenizer, description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.description = description
        self.num_querys = 256
       
        #self.box_tokens = ["<bx_start>", "<bx_end>"]

        if mode == "train":
            self.data_list = load_from_disk(args.celltype_data_train_path)
        elif mode == "validation":
            self.data_list = load_from_disk(args.celltype_data_eval_path)
        elif mode == "test":
            self.data_list = load_from_disk(args.celltype_data_test_path)

        with open(self.args.token_to_gene_dict_path, 'rb') as file:
           self.token_to_gene_dict = pickle.load(file)


        self.question_list = [
            "I randomly masked some genes. Could you predict these genes in order for me?",
            "Some genes have been masked. Predict the masked genes in sequence, please.",
            "I have hidden some genes. What are the masked genes, listed in order?",
            "A few genes are masked. Fill in the missing genes in the correct order.",
            "Some genes are masked. Predict them in the right sequence.",
            "I masked certain genes. What are the masked genes, listed in order?",
            "A portion of the genes are masked. Predict them sequentially.",
            "Some genes are missing. Provide the masked genes in the correct order.",
            "I randomly masked a few genes. Predict them in sequence.",
            "Some gene expressions are masked. List the masked genes in order.",
            "A few genes are hidden. Predict the masked genes in the correct order.",
            "I masked some genes randomly. Predict them in sequence.",
            "Some genes are masked. What are the masked genes, listed in the right order?",
            "A few genes are missing. Predict them sequentially.",
            "I randomly masked some genes. Provide the missing genes in order.",
            "Some genes are masked. Predict them in the correct sequence.",
            "A portion of the genes are hidden. Predict them in order.",
            "I masked some genes. Predict the masked genes in sequence.",
            "Some genes are missing. List the masked genes in the right order.",
            "Here are some masked genes. Predict them in order.",
            "The following genes are masked. Predict them sequentially.",
            "I've masked a few genes. Predict the missing genes in sequence.",
            "Some genes are hidden. Predict them in the correct order.",
            "A few genes are masked. Guess the masked genes in order.",
            "I randomly masked some genes. Tell me the missing genes in sequence.",
            "Some genes are masked. Predict what they are in order.",
            "I masked some genes. Predict the hidden genes in order.",
            "Some genes are hidden. Predict them in sequence.",
            "A few genes are masked. Predict the missing genes in order.",
            "I masked some genes. Predict the hidden genes in the correct sequence."

         
        ]

        self.answer_list = [
            "The predicted genes in order are {}.",
            "Based on the data, the genes in order are {}.",
            "The genes in sequence are {}.",
            "The correct order of the genes is {}.",
            "The predicted sequence of genes is {}.",
            "Here are the predicted genes in order: {}.",
            "The genes, listed in order, are {}.",
            "The predicted order of the genes is {}.",
            "The genes in the correct sequence are {}.",
            "The missing genes, in order, are {}.",
            "The predicted genes, in sequence, are {}.",
            "The genes are predicted to be in the following order: {}.",
            "The masked genes, in order, are {}.",
            "The genes are {} when listed in order.",
            "The predicted genes, in the right sequence, are {}.",
            "The genes in the expected order are {}.",
            "The genes, predicted in order, are {}.",
            "The genes are {} in the correct sequence.",
            "The predicted genes, in the proper order, are {}.",
            "The genes, in the predicted order, are {}."]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]
            gene_input_ids = data['input_ids']  # 基因表达数据的输入 ID
            gene_input_ids = torch.tensor(gene_input_ids)

            # Create a mask for 5% of the gene_input_ids
            mask = torch.rand(gene_input_ids.shape) < 0.05
            masked_gene_input_ids = gene_input_ids.clone()
            masked_gene_input_ids[mask] = 1  # geneformer token id

            # Generate the answer (the masked values)
            answer = gene_input_ids[mask].tolist()
            answer =  answer[:5] 

            # 将 token ID 转换为基因名字
            answer_genes = [self.token_to_gene_dict.get(token_id, f"Unknown_{token_id}") for token_id in answer]
            
            

            # Convert the answer to a string
            answer_str = ', '.join(map(str, answer_genes))


            # Randomly select a question and answer template
            #question_template = random.choice(self.question_list)
            #answer_template = random.choice(self.answer_list)
            question_template = "I masked some genes. Predict the top 3 hidden genes in the correct sequence, please."
            answer_template = "The top3 genes, listed in order, are {}."

            # Format the question and answer、
            self.image_tokens = "<im_patch>" * self.num_querys
            question = self.image_tokens + ' ' + question_template
            answer = answer_template.format(answer_str)


            # Tokenize the question and answer
            text_tensor = self.tokenizer(
                question + ' ' + answer, max_length=self.args.max_seq, truncation=True, padding="max_length",
                return_tensors="pt"
            )
            text_input_ids = text_tensor["input_ids"][0]  # 文本编码后的输入 ID
            attention_mask = text_tensor["attention_mask"][0]

            # Ensure the text_input_ids ends with the EOS token if it's truncated
            valid_len = torch.sum(attention_mask)
            if valid_len < len(text_input_ids):
                text_input_ids[valid_len] = self.tokenizer.eos_token_id

            # Tokenize the question separately to determine the question length
            question_tensor = self.tokenizer(
                question, max_length=self.args.max_seq, truncation=True, padding="max_length", return_tensors="pt"
            )
            question_len = torch.sum(question_tensor["attention_mask"][0])

            # Create the label tensor
            label = text_input_ids.clone()
            label[label == self.tokenizer.pad_token_id] = -100
            label[:question_len] = -100

            # Prepare the return dictionary
            ret = {
                'image': masked_gene_input_ids,  # 掩码后的基因表达数据
                'label': label,  
                'input_id': text_input_ids,  # 文本编码后的输入 ID
                'attention_mask': attention_mask,  # 文本的注意力掩码
                'question': question,  # 问题文本
                'answer': answer,  # 回答文本
                'question_type': "MaskedGene",  # 问题类型
            }

            return ret

            #except Exception as e:
            #   print(f"Error in __getitem__ at index {idx}: {e}")
            #   idx = random.randint(0, len(self.data_list) - 1)


class MaskedGeneOrderDataset(Dataset):
    def __init__(self, args, tokenizer, description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.description = description
        self.num_querys = 256

        # Load dataset
        if mode == "train":
            self.data_list = load_from_disk(args.celltype_data_train_path)
        elif mode == "validation":
            self.data_list = load_from_disk(args.celltype_data_eval_path)
        elif mode == "test":
            self.data_list = load_from_disk(args.celltype_data_test_path)

        self.species_to_dict = {
            'human': args.token_to_gene_dict_human_path,
            'macaca': args.token_to_gene_dict_macaca_path,
            'mouse': args.token_to_gene_dict_mouse_path,
            'marmoset': args.token_to_gene_dict_marmoset_path
        }
        
        # Load all token-to-gene dictionaries
        self.token_to_gene_dicts = {}
        for species, path in self.species_to_dict.items():
            with open(path, 'rb') as file:
                self.token_to_gene_dicts[species] = pickle.load(file)


        # Fixed templates (consistent with original)
        #self.question_template = "I selected some genes where the order represents expression levels of the cell. These shuffled genes are: {}. Predict their correct expression order."
        #self.answer_template = "The correct expression order from high to low is: {}."

        
        self.question_list = [
            "I selected some genes where the order represents expression levels of the cell. These shuffled genes are: {}. Predict their correct expression order.",
            "These genes represent cellular expression levels but have been randomly ordered: {}. Determine their proper expression ranking.",
            "The following genes (currently shuffled) reflect a cell's expression profile: {}. Predict their true high-to-low expression sequence.",
            "I've randomized the order of these cell expression-related genes: {}. Restore their correct expression hierarchy.",
            "These genes, representing cellular expression levels, are out of order: {}. Sort them by their actual expression intensity.",
            "The expression-level ranking of these cellular genes was scrambled: {}. Reconstruct their proper order from highest to lowest.",
            "Below are genes showing cellular expression patterns in mixed order: {}. Arrange them by decreasing expression strength.",
            "These cellular genes (currently disordered) should reflect expression levels: {}. What is their correct expression sequence?",
            "The expression order of these cell-related genes was shuffled: {}. Predict their original ranking from most to least expressed.",
            "Here are genes representing cellular expression in random sequence: {}. Determine their true expression order.",
            "I've disrupted the expression-level order of these cell genes: {}. Restore their proper ranking."]
            

        self.answer_list = [
            "The proper cellular expression ranking (high to low) is: {}.",
            "From highest to lowest cellular expression, the order should be: {}.",
            "The correct sequence by cell expression levels is: {}.",
            "Ranked by decreasing cellular expression: {}.",
            "It is: {}.",
            "The true expression hierarchy in cells is: {}.",
            "From most to least expressed in cells: {}.",
            "The cellular expression-sorted order is: {}."
           ]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        gene_input_ids = torch.tensor(data['input_ids'])  # Original gene IDs (pre-sorted by expression)
        # Get species for this sample
        species = data['species'].lower()  # ensure lowercase for dictionary lookup
        
        # 1. Randomly select 5 unique genes (no duplicates)
        if len(gene_input_ids) >= 5:
            selected_indices = torch.randperm(len(gene_input_ids))[:5]
        else:
            # Fallback if less than 5 genes (unlikely per requirements)
            selected_indices = torch.arange(len(gene_input_ids))
        
        selected_genes = gene_input_ids[selected_indices].tolist()
        

        masked_gene_input_ids = gene_input_ids.clone()
        masked_gene_input_ids[selected_indices] = 1  # geneformer token id

        # 2. Convert token IDs to gene names using species-specific dictionary
        gene_names = [self.token_to_gene_dicts[species].get(gene_id, f"Unknown_{gene_id}") for gene_id in selected_genes]
        
        # 3. Shuffle for question display
        shuffled_indices = torch.randperm(len(selected_indices))
        shuffled_genes = [gene_names[i] for i in shuffled_indices]
        
        # 4. Sort by original expression for answer
        sorted_indices = sorted(range(len(selected_indices)), key=lambda i: selected_indices[i])
        answer_genes = [gene_names[i] for i in sorted_indices]
        

        
        question_template = random.choice(self.question_list)
        #print("this is question temple",question_temple)
        #print("this is question",question)
        answer_template = random.choice(self.answer_list)
        #print("this is question temple",question_temple)
        try:
            question_text = question_template.format(", ".join(shuffled_genes[:-1]) + " and " + shuffled_genes[-1])
        except:
            print(f"Warning: Empty gene list encountered (idx: {idx})")
            print("Current shuffled_genes:", shuffled_genes)  # 打印当前值
            breakpoint()  # 进入调试器（必须加括号0
            question_text = question_template.format(", ".join(shuffled_genes))  # Fallback value

        question_text = question_template.format(", ".join(shuffled_genes[:-1]) + " and " + shuffled_genes[-1])
        answer_text = answer_template.format( ", ".join(answer_genes[:-1]) + " and " + answer_genes[-1])
        
        

        # Add image tokens (consistent with original)
        self.image_tokens = "<im_patch>" * self.num_querys
        #question = f"{self.image_tokens} {question_text}"
        

        question = self.image_tokens + ' ' + question_text
        
         # Tokenize the question and answer
        text_tensor = self.tokenizer(
            question + ' ' + answer_text, max_length=self.args.max_seq, truncation=True, padding="max_length",
            return_tensors="pt"
            )
        text_input_ids = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]

        # Ensure EOS token if truncated (identical to original)
        valid_len = torch.sum(attention_mask)
        if valid_len < len(text_input_ids):
            text_input_ids[valid_len] = self.tokenizer.eos_token_id

        # Get question length (identical to original)
        question_tensor = self.tokenizer(
            question,
            max_length=self.args.max_seq,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        question_len = torch.sum(question_tensor["attention_mask"][0])

        # Create labels (identical to original)
        label = text_input_ids.clone()
        label[label == self.tokenizer.pad_token_id] = -100
        label[:question_len] = -100

        # Return dictionary (consistent structure with original)
        ret = {
            'image': masked_gene_input_ids,  
            'label': label,
            'input_id': text_input_ids,
            'attention_mask': attention_mask,
            'question': question,
            'answer': answer_text,
            'question_type': "GeneOrderPrediction",  # Updated type
        }

        return ret

class SpatialMaskGeneDataset(Dataset):
    def __init__(self, args, tokenizer, description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.description = description
        self.num_querys = 256
      
       
        #self.box_tokens = ["<bx_start>", "<bx_end>"]

        if mode == "train":
            self.data_list = load_from_disk(args.spatial_data_train_path)
        elif mode == "validation":
            self.data_list = load_from_disk(args.spatial_data_eval_path)
        elif mode == "test":
            self.data_list = load_from_disk(args.spatial_data_test_path)

        
        self.species_to_dict = {
            'human': args.token_to_gene_dict_human_path,
            'macaca': args.token_to_gene_dict_macaca_path,
            'mouse': args.token_to_gene_dict_mouse_path,
            'marmoset': args.token_to_gene_dict_marmoset_path
        }
        
        # Load all token-to-gene dictionaries
        self.token_to_gene_dicts = {}
        for species, path in self.species_to_dict.items():
            with open(path, 'rb') as file:
                self.token_to_gene_dicts[species] = pickle.load(file)


        #self.label_to_idx = {d['cell_label']: i for i, d in enumerate(self.data_list)}
        #self.label_to_idx = pickle.load(open("/fs-computility/mabasic/weiran/brain/label_to_idx.pkl", "rb"))
        unique_labels = self.data_list.unique('cell_label')
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        self.question_list = [

          "This is the average expression value of the neighboring spots around a spatial spot. Based on the expression of the neighboring spots, please generate the top 5 genes of this spot in the order of expression levels.",
          "Here are the average expression values from adjacent spots surrounding a spatial location. Using this neighboring expression data, predict the top 5 most highly expressed genes at this spot, ranked by expression level.",
          "The following represents mean expression values from spots surrounding a central spatial location. From this neighboring expression profile, determine the central spot's top 5 genes ordered by expression intensity.",
          "These values show average gene expression in the vicinity of a spatial spot. Based on this local expression pattern, what are this spot's 5 most abundant transcripts, listed by decreasing expression levels?",
          "Presented is the averaged expression profile from a spot's immediate neighborhood. Using this surrounding expression data, identify the spot's top 5 expressed genes ranked from highest to lowest.",
          "This data reflects the average expression pattern in the area surrounding a spatial location. Based on these neighboring expression values, provide the spot's 5 most active genes in order of expression magnitude.",

        ]

        self.answer_list = [
            "The top 5 genes are {}.",
            "Ranked from highest to lowest, the top 5 expressed genes are {}.",
            "In descending order of expression, the top 5 genes are {}.",
            "The five most highly expressed genes, in order, are {}.",
            "From most to least expressed, the top 5 genes are {}.",
            "The five genes with strongest expression, ranked accordingly, are {}."
           ]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            gene_input_ids = data['input_ids']
            gene_input_ids = torch.tensor(gene_input_ids)
            species = data['species'].lower()  # ensure lowercase for dictionary lookup

            # Generate the answer (the masked values)
            answer = gene_input_ids.tolist()
            answer =  answer[:5] 

            # 将 token ID 转换为基因名字
            #answer = [self.token_to_gene_dict.get(token_id, f"Unknown_{token_id}") for token_id in answer]
             # 2. Convert token IDs to gene names using species-specific dictionary
            answer = [self.token_to_gene_dicts[species].get(gene_id, f"Unknown_{gene_id}") for gene_id in answer]
        
            #answer_str = ', '.join(map(str, answer))
            answer_str = ", ".join(answer[:-1]) + " and " + answer[-1]
            #cell_type = data['celltype'] # 

            #soma_id = data['soma_joinid']
            #batch_labels = data['domain_labels']
            nearest_labels = data['nearest_10_cell_labels']
            nearest_input_ids = []
            
            for label in nearest_labels:
                nearest_idx = self.label_to_idx.get(label, None)
                if nearest_idx is not None:
                    # 获取input_ids并转为tensor
                    nearest_input_ids.append(torch.tensor(self.data_list[nearest_idx]['input_ids']))
                else:
                    # 使用与gene_input_ids相同形状的零张量
                    nearest_input_ids.append(torch.zeros_like(gene_input_ids))

            # 使用pad_sequence自动填充对齐（替代torch.stack）
            if len(nearest_input_ids) > 0:
                nearest_input_ids = pad_sequence(
                    nearest_input_ids,
                    batch_first=True,
                    padding_value=0  # 填充值为0
                )
            else:
                # 处理空列表情况
                nearest_input_ids = torch.zeros((0, len(gene_input_ids)), dtype=gene_input_ids.dtype)
    

            self.image_tokens = "<im_patch>" * self.num_querys
            # 添加调试信息
            
            #print(self.description)
            #print(self.question_list)

            #try:
            #question_temple = random.choice(self.question_list)
           
            #question_template = "This is the average expression value of the neighboring spots around a spatial spot. Based on the expression of the neighboring spots, please generate the top 5 genes of this spot in the order of expression levels."
            #answer_template = "The top5 genes, listed in order, are {}."

            question_template = random.choice(self.question_list)
            #print("this is question temple",question_temple)
            #print("this is question",question)
            answer_template = random.choice(self.answer_list)
            #print("this is question temple",question_temple)
            question = self.image_tokens + ' ' +   question_template
            #print("this is question",question)
            answer = answer_template.format(answer_str)

            text_tensor = self.tokenizer(
                question + ' ' + answer, max_length=self.args.max_seq, truncation=True, padding="max_length",
                return_tensors="pt"
            )
            input_id = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]

            valid_len = torch.sum(attention_mask)
            if valid_len < len(input_id):
                input_id[valid_len] = self.tokenizer.eos_token_id

            question_tensor = self.tokenizer(
                question, max_length=self.args.max_seq, truncation=True, padding="max_length", return_tensors="pt"
            )
            question_len = torch.sum(question_tensor["attention_mask"][0])

            label = input_id.clone()
            label[label == self.tokenizer.pad_token_id] = -100
            label[:question_len] = -100
            #print(self.description)

            ret = {
                'image': nearest_input_ids,
                'input_id': input_id,
                'label': label,
                'attention_mask': attention_mask,
                'question': question,
                'answer': answer,
                'question_type': "SpatialMask",
            }
            return ret

            #except Exception as e:
            #   print(f"Error in __getitem__ at index {idx}: {e}")
            #   idx = random.randint(0, len(self.data_list) - 1)

class UniDatasets(Dataset):
    def __init__(self, args, tokenizer, description='False',mode='train'):
        super(UniDatasets, self).__init__()
        self.ds_list = [
            #MaskedGeneDataset(args, tokenizer,description,mode),
            CellTypeDataset(args, tokenizer,description,mode),
            SubclassDataset(args, tokenizer,description,mode),
            TissueTypeDataset(args, tokenizer,description,mode),
            DevelopmentStageDataset(args, tokenizer,description,mode),
            DiseaseDataset(args, tokenizer,description,mode),
            MaskedGeneOrderDataset(args, tokenizer,description,mode),
            SpatialMaskGeneDataset(args, tokenizer,description,mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

