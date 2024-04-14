import sys


sys.path.append("./LLaVA")


from PIL import Image

import torch
import os
import numpy as np
from torchvision import transforms
import json
import argparse
from self_filter_model import (
    LlavaLlamaForCausalLM_SelfFilter_CLIP,
    LlavaLlamaForCausalLM_SelfFilter_Scores,
)
import tqdm
from transformers.modeling_utils import load_sharded_checkpoint


def load_stage1_model(
    model_path, feature_extractor_setting, device_map="auto", device="cuda", **kwargs
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    kwargs["torch_dtype"] = torch.float16

    # note that we do not need vision tower here, and it is not loaded.
    if feature_extractor_setting == "clip":
        model = LlavaLlamaForCausalLM_SelfFilter_CLIP.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
    elif feature_extractor_setting == "scores":
        model = LlavaLlamaForCausalLM_SelfFilter_Scores.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
    else:
        print("Unknown feature extractor setting: ", feature_extractor_setting)
        raise NotImplementedError
    # model.init_score_net(feature_extractor_setting)

    # load the weights for score net
    # load_sharded_checkpoint(model,model_path,strict=False)

    # return model.to(device)
    return model


def load_scores(score_names):
    def norm_scores(score_dict: dict):
        min_score = min(score_dict.values())
        max_score = max(score_dict.values())
        normed_score_dict = {
            i[0]: (i[1] - min_score) / (max_score - min_score) * 2 - 1
            for i in score_dict.items()
        }
        return normed_score_dict

    score_dicts = []

    for score_name in score_names:
        with open(score_name, "r") as f:
            score_dict = json.load(f)
            score_dicts.append(norm_scores(score_dict))

    return score_dicts


def produce_scores_difficulty(model, save_path: str):
    difficulty_dict = {}

    score_dicts = [
        "data/scores/llava_imagereward.json",
        "data/scores/llava_clipscore.json",
        "data/scores/gpt-3.5-turbo-1106/processed_score.json",
    ]

    for unique_idx in score_dicts[0]:
        scores = [[score_dict[str(unique_idx)] for score_dict in score_dicts]]
        scores = torch.tensor(scores).cuda().half()
        difficulty_dict[unique_idx] = -model.predict_weights(scores).item()

    with open(save_path, "w") as f:
        json.dump(difficulty_dict, f)

    print("Scores difficulty generated and saved.")

    return difficulty_dict


def produce_clip_difficulty(model, save_path: str):
    difficulty_dict = {}
    clip_feat = torch.load("data/scores/llava_clip_feature.pt")

    for unique_idx in clip_feat:
        dtype = model.get_score_net_dtype()
        scores = clip_feat[unique_idx].unsqueeze(dim=0).cuda().to(dtype=dtype)
        difficulty_dict[unique_idx] = -model.predict_weights(scores).item()

    with open(save_path, "w") as f:
        json.dump(difficulty_dict, f)

    print("CLIP difficulty generated and saved")

    return difficulty_dict


def get_difficulty_score(
    model_path: str, feature_extractor_setting: str, save_path: str
):
    if os.path.exists(save_path):
        print("Difficulty already exists, generation skipped.")
        with open(save_path, "r") as f:
            difficulty_dict = json.load(f)
        return difficulty_dict
    print("Loading stage 1 model...", flush=True)
    model = load_stage1_model(model_path, feature_extractor_setting)
    print("Model loaded.", flush=True)

    if feature_extractor_setting == "scores":
        return produce_scores_difficulty(model, save_path)
    else:
        return produce_clip_difficulty(model, save_path)


def dist_filter(
    raw_annotation_path, difficulty_dict, filter_num, save_path, gamma=1, k_nearest=10
):
    if os.path.exists(save_path):
        print("Filtered annotation already exists.")
        return
    with open(raw_annotation_path, "r") as f:
        raw_annotation = json.load(f)
    new_annotation = []

    feat_dict = torch.load("data/scores/llava_clip_feature.pt")
    feat_len = len(feat_dict)
    feat_matrix = torch.stack(
        [feat_dict[str(i)].cuda() for i in range(feat_len)], dim=0
    )
    feat_matrix_norm = torch.norm(feat_matrix, dim=-1, keepdim=False)

    for i in tqdm.tqdm(range(filter_num)):
        lst = sorted(difficulty_dict.items(), key=lambda x: x[1], reverse=True)

        unique_idx, difficulty = lst[0]

        example = raw_annotation[int(unique_idx)]
        # assert example['unique_idx']==int(unique_idx)

        example.pop("unique_idx")
        new_annotation.append(example)

        difficulty_dict.pop(unique_idx)

        tgt_feat = feat_matrix[int(unique_idx)].unsqueeze(dim=0)
        tgt_norm = feat_matrix_norm[int(unique_idx)].unsqueeze(dim=0)

        sims = (feat_matrix * tgt_feat).sum(dim=-1) / feat_matrix_norm / tgt_norm

        sorted_sim, indices = torch.sort(sims, descending=True)

        success_cnt = 0

        for j in range(len(difficulty_dict)):
            if success_cnt >= k_nearest:
                break

            cur_unique_idx = str(indices[j].item())

            if cur_unique_idx not in difficulty_dict:
                continue

            cur_sim = sorted_sim[j].item()
            penalty = difficulty * (cur_sim**2) * gamma
            difficulty_dict[cur_unique_idx] -= penalty
            success_cnt += 1

        assert success_cnt == k_nearest

    with open(save_path, "w") as f:
        json.dump(new_annotation, f)

    print("Annotation filtered and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_model_path", type=str)
    parser.add_argument(
        "--feature_extractor_setting", type=str, choices=["scores", "clip"]
    )

    parser.add_argument("--result_dir", type=str, default="./data/results")
    parser.add_argument("--difficulty_save_name", type=str)

    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--k_nearest", type=int, default=10)
    parser.add_argument(
        "--raw_annotation_path",
        type=str,
        default="data/llava_instruct_158k_add_idx.json",
    )
    parser.add_argument("--filtered_annotation_save_path", type=str)
    parser.add_argument("--filter_num", type=int)
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    difficulty_dict = get_difficulty_score(
        args.stage1_model_path,
        args.feature_extractor_setting,
        os.path.join(args.result_dir, args.difficulty_save_name),
    )
    
    # exit(0)
    dist_filter(
        args.raw_annotation_path,
        difficulty_dict,
        args.filter_num,
        args.filtered_annotation_save_path,
        args.gamma,
        args.k_nearest,
    )
