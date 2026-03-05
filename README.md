# scMOBA: A conversational single-cell multi-omics foundation model of the brain
Single-cell and spatial multi-omics are revolutionizing our understanding of developmental, aging, and diseased brains, yet integration across modalities and species remains highly challenging. We developed a conversational single-cell multi-omics foundation model of the brain (scMOBA) that integrates a large language model with a gene encoder and a cross-attention projector. By pretraining on ~133 million cells via a novel Feature–Question–Answer paradigm, scMOBA enables unified modeling and prediction across various omics, species, and biological tasks, allowing for direct zero-shot inference without task-specific fine-tuning. scMOBA achieved state-of-the-art performance in cell typing, multi-omics and cross-species integration. We further demonstrated its use in identifying evolutionarily conserved and species-specific cell types, reconstructing developmental trajectories, constructing aging clocks, and predicting disease states. Together, scMOBA provides a unified and extensible framework for integrated and interpretable multi-omics analysis across species and biological contexts.
## Model Architecture

scMOBA integrates a **gene feature encoder**, a **cross-attention projector**, and a **large language model (LLM)** to enable conversational reasoning over single-cell multi-omics data.

The architecture consists of three main components:
1. **Gene Feature Encoder**
   
   Encodes gene expression features from single-cell or spatial omics data into dense representations. We initialize the encoder using the pretrained **GeneFormer** model.
3. **Cross-Attention Projector**
   
   Aligns gene feature representations with the token space of the language model.  
   This module bridges biological features and language tokens through cross-modal attention.
5. **Large Language Model (LLM)**

   The LLM Inference backbone of scMOBA, initialized with **Llama-3.2-1B**, enabling conversational biological inference and downstream task prediction.
## Training Tasks
scMOBA is trained with a unified **Feature–Question–Answer (FQA)** framework across multiple biological tasks.
The training dataset includes:
- Cell type prediction
- Cell subclass classification
- Tissue identification
- Developmental stage inference
- Disease prediction
- Masked gene prediction
- Gene expression ranking
- Spatial transcriptomics reasoning

Each task is converted into an instruction format:
Feature (gene expression) + Question → Answer
The dataset construction and task definitions are implemented in  
`src/datasets/muti_dataset.py`

All tasks are combined through a unified dataset loader implemented in
`UniDatasets` (see `src/datasets//muti_dataset.py`).
## Installation
```bash
git clone https://github.com/Sherryweiran/scMOBA.git
cd scMOBA
conda env create -f scmoba_environment.yml
conda activate scmoba
```
## Training
### Pretrained Initialization Weights

To train **scMOBA**, we initialize the model with pretrained weights to achieve better performance and faster convergence.
#### Gene Feature Encoder
We initialize the gene feature encoder with **Geneformer**.
Please download the initialization weight from the Geneformer repository:
https://huggingface.co/ctheodoris/Geneformer/tree/main/Geneformer-V1-10M

#### LLM
The language backbone is initialized with **Llama-3.2-1B**.
You can download it from HuggingFace:
https://huggingface.co/meta-llama/Llama-3.2-1B

### Training
Our training consists of two steps. 
- **Step 1**: [Alignment](#step-1-alignment)
- **Step 2**: [Instruction Pretraining](#step-2-instruction-pretraining)
#### Configuration
We suggest using `accelerate` to train. It was developed by Hugging Face 
and conveniently supports common training strategies such as distributed training, mixed precision, etc.
It should be configured on first use:
```bash
accelerate config
```
Please follow the configuration guide and we can choose the appropriate training strategy. 
We recommend using bf16 for acceleration.

If you don't know how to configure it, we provide a simple configuration `default_config.yaml` for your reference.
<details>
<summary>default_config.yaml</summary>

```bash
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
</details>

#### Step 1: Alignment
We should align gene feature and language with feature-text data, that is, only train projector and freeze the gene feature encoder and LLM. 

Please update LLM path `--llm_path` and gene feature encoder path `--bert_path`, respectively.
Update `--output_dir` to specify the output path of the model.
Then run the script by:
```bash
sh script/train_step1.sh
```
#### Step 2: Instruction Pretraining
Single-cell instruction tuning through multi-task data of FQA (feature-question-answer pairs),  
During this stage, LoRA fine-tuning is applied to the LLM, while the gene feature encoder and the projector are also jointly trained.

Please update LLM path `--llm_path` and gene feature encoder path `--bert_path`, respectively.
Update `--output_dir` to specify the output path of the model.
Then run the script by:
```bash
sh script/train_step2.sh
```

### Evaluation
We can directly evaluate each task by running:
```bash
python src/eval/eval_TASK.py
```


## Citation
If our project are helpful to you, please consider citing:

```BibTeX
@article{Wei2025scMOBA,
  author = {Wei, Ran and Zhang, Ziyao and Sun, Jianle and Sun, Yongkang and Meng, Juan and Zheng, Peng and Liang, Chaoqi and Meng, Fanyi and Ouyang, Wanli and Bai, Lei and Ye, Peng and Sun, Yidi},
  title = {scMOBA: A conversational single-cell Multi-Omics Brain Agent across species},
  journal = {bioRxiv},
  year = {2025},
  doi = {10.64898/2025.12.01.691565},
  url = {https://www.biorxiv.org/content/early/2025/12/02/2025.12.01.691565}
}
```

## Acknowledgement
We gratefully acknowledge the open-source projects that inspired and supported this work, including 
[GeneFormer](https://github.com/jkobject/geneformer) and 
[LLaMA](https://github.com/meta-llama/llama). 
We thank the respective teams for making their code and models publicly available.
