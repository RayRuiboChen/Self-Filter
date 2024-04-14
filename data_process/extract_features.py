import json
import os
import argparse
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import tqdm
import torch


def extract_clip_score(vision_tower, text_data_path, image_dir, save_dir):
    res_filename = os.path.join(save_dir, "llava_clipscore.json")

    if os.path.exists(res_filename):
        print("CLIP score already exists.")
        return

    print("Extracting CLIP score ...")
    model = CLIPModel.from_pretrained(vision_tower).cuda()
    processor = CLIPProcessor.from_pretrained(vision_tower)

    clipscores = {}

    with open(text_data_path, "r") as f:
        text_data = json.load(f)

    model.eval()
    with torch.no_grad():
        for unique_idx in tqdm.tqdm(text_data):
            image_filename = text_data[unique_idx]["image"]
            text = text_data[unique_idx]["text"]
            filename = os.path.join(image_dir, image_filename)
            image = Image.open(filename).convert("RGB")

    text_inputs = processor(text=text, return_tensors="pt", truncation=True)
    image_inputs = processor(images=image, return_tensors="pt")

    text_inputs = {i[0]: i[1].cuda() for i in text_inputs.items()}
    image_inputs = {i[0]: i[1].cuda() for i in image_inputs.items()}

    text_features = model.get_text_features(**text_inputs)
    image_features = model.get_image_features(**image_inputs)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    clip_score = text_features @ image_features.T
    clipscores[unique_idx] = clip_score.item()

    with open(res_filename, "w") as f:
        json.dump(clipscores, f)
    print("CLIP score extracted and saved.")


def extract_imagereward_score(text_data_path, image_dir, save_dir):
    import ImageReward as RM

    res_filename = os.path.join(save_dir, "llava_imagereward.json")

    if os.path.exists(res_filename):
        print("ImageReward Score already exists")
        return

    print("Extracting ImageReward score ...")

    model = RM.load("ImageReward-v1.0").cuda()
    reward_dict = {}

    with open(text_data_path, "r") as f:
        text_data = json.load(f)

    model.eval()
    with torch.no_grad():
        for unique_idx in tqdm.tqdm(text_data):
            image_filename = text_data[unique_idx]["image"]
            image_filename = os.path.join(image_dir, image_filename)
            text = text_data[unique_idx]["text"]
            rewards = model.score(text, [image_filename])
            reward_dict[unique_idx] = rewards

    with open(res_filename, "w") as f:
        json.dump(reward_dict, f)
    print("ImageReward Score extracted and saved.")


def extract_clip_features(vision_tower, text_data_path, image_dir, save_dir):
    save_path = os.path.join(save_dir, "llava_clip_feature.pt")
    if os.path.exists(save_path):
        print("CLIP features already exist.")
        return

    print("Extracting CLIP features ...")
    model = CLIPModel.from_pretrained(vision_tower).cuda()
    processor = CLIPProcessor.from_pretrained(vision_tower)

    with open(text_data_path, "r") as f:
        text_data = json.load(f)

    tot_feat = {}
    model.eval()
    with torch.no_grad():
        for unique_idx in tqdm.tqdm(text_data):
            image_filename = text_data[unique_idx]["image"]
            text = text_data[unique_idx]["text"]

            filename = os.path.join(image_dir, image_filename)
            image = Image.open(filename).convert("RGB")
            text_inputs = processor(text=text, return_tensors="pt", truncation=True)
            image_inputs = processor(images=image, return_tensors="pt")

            text_inputs = {i[0]: i[1].cuda() for i in text_inputs.items()}
            image_inputs = {i[0]: i[1].cuda() for i in image_inputs.items()}

            text_features = model.get_text_features(**text_inputs).squeeze(dim=0)
            image_features = model.get_image_features(**image_inputs).squeeze(dim=0)

            image_features = image_features / image_features.norm(dim=-1)
            text_features = text_features / text_features.norm(dim=-1)

            sample_feature = torch.concatenate([image_features, text_features], dim=0)
            tot_feat[unique_idx] = sample_feature.cpu()

    torch.save(tot_feat, save_path)
    print("CLIP features extracted and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_extractor_type",
        type=str,
        choices=["clip_score", "imagereward_score", "clip_features"],
    )

    parser.add_argument("--text_data_path", type=str, default="./data/text_data.json")
    parser.add_argument("--image_dir", type=str, default="./data/coco/train2017")
    parser.add_argument("--save_dir", type=str, default="./data/scores")

    parser.add_argument(
        "--vision_tower", type=str, default="openai/clip-vit-large-patch14"
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.feature_extractor_type == "clip_score":
        extract_clip_score(
            args.vision_tower, args.text_data_path, args.image_dir, args.save_dir
        )
    elif args.feature_extractor_type == "imagereward_score":
        extract_imagereward_score(args.text_data_path, args.image_dir, args.save_dir)
    elif args.feature_extractor_type == "clip_features":
        extract_clip_features(
            args.vision_tower, args.text_data_path, args.image_dir, args.save_dir
        )
    else:
        print("Unknown feature extractor type: ", args.feature_extractor_type)
        raise NotImplementedError
