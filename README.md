# Multi-Modal Large Language Model Enables Protein Function Prediction
<div align="center">

[![Data](https://img.shields.io/badge/Data-4d5eff?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor)](https://huggingface.co/datasets/mignonjia/ProteinChatQA)
[![Model](https://img.shields.io/badge/Model-4d5eff?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor)](https://huggingface.co/mignonjia/proteinglm-1b-mlm-sft)
</div>

## Examples

![Eg1](fig/example.png) 

Examples of multi-round dialogues with ProteinChat for Q9U281, Q9XZG9, and Q9LU44.

## Introduction
- ProteinChat is a versatile, multi-modal large language model designed to predict protein functions from amino acid sequences.
- ProteinChat works in a similar way as ChatGPT. Users upload a protein sequence and ask various questions about this protein. ProteinChat will answer these questions in a multi-turn, interactive manner. 
- The ProteinChat system consists of a protein encoder, a large language model (LLM), and an adaptor. The protein encoder takes a protein sequence as input and learns a representation for this protein. The adaptor transforms the protein representation produced by the protein encoder into another representation that is acceptable to the LLM. The LLM takes the representation transformed by the adaptor and users' questions about this protein as inputs and generates answers. All these components are trained end-to-end. We use [xTrimoPGLM-1B](https://arxiv.org/abs/2401.06199) as the protein encoder.
- To train ProteinChat, we designed (protein, prompt, answer) triplets from the functions and keywords from Swiss-Prot dataset, resulting in ~500k proteins and 1.5 million triplets.

![overview](fig/workflow.png)


## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and activate it via the following command (instllation time: ~ 4min)

```bash
cd ProteinChat
conda env create -f environment.yml
conda activate proteinchat
```

Verify the installation of `torch` and `torchvision` is successful by running `python -c "import torchvision; print(torchvision.__version__)"`. If it outputs the version number without any warnings or errors, then you are good to go. __If it outputs any warnings or errors__, try to uninstall `torch` by `conda uninstall pytorch torchvision torchaudio cudatoolkit` and then reinstall them following [here](https://pytorch.org/get-started/previous-versions/#v1121). You need to find the correct command according to the CUDA version your GPU driver supports (check `nvidia-smi`). 

For aarch64 machines:

```bash
cd ProteinChat
conda env create -f environment_arm.yml
conda activate proteinchat_arm

# First, verify if torch from environment_arm.yml works
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Similarly to above, if torch needs to be reinstalled, run:
# pip uninstall -y torch torchaudio torchvision
# Select your own version from https://pytorch.org/get-started/previous-versions/, such as:
# pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

**2. Data and Model**

This codebase loads [data](https://huggingface.co/datasets/mignonjia/ProteinChatQA) and [model](https://huggingface.co/mignonjia/proteinglm-1b-mlm-sft) directly from hugging face.

To access these resources, you need to set your Hugging Face token by running:

```bash
export HF_TOKEN=your_token_here
```

Alternatively, you can use `huggingface-cli login` to authenticate interactively.

### Training
**You need at least 55 GB GPU memory for the training.** 

The stage-1 training configuration file is [configs/proteinchat_stage1.yaml](configs/proteinchat_stage1.yaml). In addition, you may want to change the number of epochs and other hyper-parameters there, such as `max_epoch`, `init_lr`, `min_lr`,`warmup_steps`, `batch_size_train`. Please adjust `iters_per_epoch` so that `iters_per_epoch` * `batch_size_train` = your training set size. 

Also, set your desired output directory [here](configs/proteinchat_stage1.yaml#52).

Start stage-1 training by running 
```bash
bash finetune.sh --cfg-path configs/proteinchat_stage1.yaml
``` 

The stage-2 training configuration file is [configs/proteinchat_stage2.yaml](configs/proteinchat_stage2.yaml). Replace the `stage1_ckpt` with the checkpoint you obtained in stage 1. Similar with the previous step, you also need to replace the output directory in this file.

Start stage-2 training by running 
```bash
bash finetune.sh --cfg-path configs/proteinchat_stage2.yaml
``` 

### Evaluation

**It takes around 24 GB GPU memory for the inference.**

Modify the checkpoint paths in [configs/proteinchat_eval.yaml](configs/proteinchat_eval.yaml) to the location of your checkpoint.
Download our ProteinChat's stage1_ckpt [here](https://drive.google.com/file/d/1H-POt4e5Q5fYF59ZwfSdAJyuQiJ2rtJl/view?usp=sharing). peft_ckpt can be set empty during evaluation.
To evaluate stage-2, [this parameter](configs/proteinchat_eval.yaml#L6) needs to be set False.

Evaluate on entire specific-category prediction by running (~1 hour)
```bash
bash demo.sh
``` 


## Acknowledgement

+ [xTrimoPGLM](https://arxiv.org/abs/2401.06199)
+ [ESM](https://github.com/facebookresearch/esm)
+ [MiniGPT-4](https://minigpt-4.github.io/) 
+ [Lavis](https://github.com/salesforce/LAVIS)
+ [Vicuna](https://github.com/lm-sys/FastChat)


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
