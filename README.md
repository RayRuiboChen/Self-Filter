<h1 align="center">Your Vision-Language Model Itself Is a Strong Filter: Towards High-Quality Instruction Tuning with Data Selection</h1>
<h4 align="center"> Ruibo Chen, Yihan Wu, Lichang Chen, Guodong Liu, Qi He, Tianyi Xiong, Chenxi Liu, Junfeng Guo, Heng Huang</h4>




## Introduction
### [[Paper]](https://arxiv.org/abs/2402.12501) ACL Findings, 2024



<p align="center">
<img src=images/multimodalfiltering_v2.jpg  width="80%" height="60%">
</p>

We introduce Self-Filter and demonstrate that vision-language models do not necessarily require a large number of data. A small amount of high-quality data is sufficient for successful instruction tuning.

Our method leverage large vision language models themselves as filters for instruction-finetuning, and does not require additional pre-defined evaluation tasks or surrogate models. It makes no assumptions about downstream tasks, thereby preserving the model’s generalization capabilities.


## Selected Samples & Model Weights

You can download our selected samples and predicted difficulty scores [here](https://huggingface.co/datasets/RayRuiboChen/Self-Filter-LLaVA-25K).

Stage 1 & Stage 2 models can be downloaded from:

| Model Name    | Feature Extractors Setting |   Training Data  |                                                                 Checkpoint                                                                 |
|---------------|----------------------------|:----------------:|:------------------------------------------------------------------------------------------------------------------------------------------:|
| Stage1-CLIP   | CLIP                       |   full dataset   |   [RayRuiboChen/LLaVA-Self-Filter-Stage1-CLIP](https://huggingface.co/RayRuiboChen/LLaVA-Self-Filter-Stage1-CLIP)   |
| Stage2-CLIP   | CLIP                       | 25K instructions |   [RayRuiboChen/LLaVA-Self-Filter-Stage2-CLIP](https://huggingface.co/RayRuiboChen/LLaVA-Self-Filter-Stage2-CLIP)   |
| Stage1-Scores | Scores                     |   full dataset   | [RayRuiboChen/LLaVA-Self-Filter-Stage1-Scores](https://huggingface.co/RayRuiboChen/LLaVA-Self-Filter-Stage1-Scores) |
| Stage2-Scores | Scores                     | 25k instructions | [RayRuiboChen/LLaVA-Self-Filter-Stage2-Scores](https://huggingface.co/RayRuiboChen/LLaVA-Self-Filter-Stage2-Scores) |


## Installation

### 1. Prepare the Environment
Please first install LLaVA：

```
cd Self-Filter
git clone https://github.com/haotian-liu/LLaVA.git
```

Then prepare the environment for LLaVA [here](https://github.com/haotian-liu/LLaVA).

### 2. Download Datasets

Download the annotation of instruction tuning data [llava_instruct_158k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json), and the image dataset COCO train2017 images from [here](https://cocodataset.org/#download).

Organize the data as follows in `./data` :

```
├── coco
│   └── train2017
└── llava_instruct_158k.json
```


## Usage

### 1. Preprocess the Dataset

We first add a unique index for each instruction in the original dataset, to better identify each sample:

```
bash scripts/preprocess.sh
```

### 2. Extract Features

(a) For the *Scores* setting, we use the CLIP score, Imagereward score and ChatGPT score as features:




For [ImageReward](https://github.com/THUDM/ImageReward), install dependency using:
```
pip install image-reward
```
then extract the scores:
```
bash scripts/extract_imagereward_score.sh
```

For CLIP score，extract the scores using:
```
bash scripts/extract_clip_score.sh
```



For ChatGPT, first prepare your openai key:
```
export OPENAI_API_KEY=Your_KEY
```
secondly, query ChatGPT to get the responses:
```
bash scripts/query_chatgpt.sh
```
Finally, parse the GPT responses to get the evaluation scores:
```
bash scripts/process_chatgpt_output.sh
```



(b) For the *CLIP* setting, we directly use the features encoded by CLIP:
```
bash scripts/extract_clip_features.sh
```

### 3. Run Self-Filter Stage 1

Please follow the instructions in [LLaVA](https://github.com/haotian-liu/LLaVA) to download the pretrained projector weights [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) and Vicuna checkpoints [here](https://github.com/lm-sys/FastChat#model-weights). 
The pre-trained models should be saved in `./checkpoints`.

Run Stage 1 with one of the feature extractor settings (*Scores* or *CLIP*):

```
bash scripts/run_stage1_scores.sh
bash scripts/run_stage1_clip.sh
```

### 4. Run Self-Filter Stage 2

Run Stage 2 using the corresponding feature extractor from Stage 1:

```
bash scripts/run_stage2_scores.sh
bash scripts/run_stage2_clip.sh
```
The filtered annotations will be saved in `./data/self_filter_25k_scores.json` or `./data/self_filter_25k_clip.json`.


### Citation

If you find our work useful for your research and applications, please consider citing:

```
@article{chen2024your,
  title={Your Vision-Language Model Itself Is a Strong Filter: Towards High-Quality Instruction Tuning with Data Selection},
  author={Chen, Ruibo and Wu, Yihan and Chen, Lichang and Liu, Guodong and He, Qi and Xiong, Tianyi and Liu, Chenxi and Guo, Junfeng and Huang, Heng},
  journal={arXiv preprint arXiv:2402.12501},
  year={2024}
}
```
