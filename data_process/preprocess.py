import json
import os
import argparse
import tqdm


def add_idx(raw_annotation_path, new_annotation_save_path):
    if os.path.exists(new_annotation_save_path):
        print("New annotation file already exists.")
    else:
        with open(raw_annotation_path, "r") as f:
            raw_annotation = json.load(f)

        new_annotation = []

        for idx, i in enumerate(raw_annotation):
            i["unique_idx"] = idx
            new_annotation.append(i)

        with open(new_annotation_save_path, "w") as f:
            json.dump(new_annotation, f)

        print("Unique index added.")


def get_input_text(conv):
    prompt = (conv[0].strip().split("\n")[-1]).replace("</s>", " ")
    prompt = "USER: " + prompt
    return prompt.strip()


def get_text_instruction(conv):
    prompt = ""

    for idx, i in enumerate(conv):
        value = i["value"]
        if idx % 2 == 0:
            assert i["from"] == "human"
            if "<image>" in value:
                value = value.replace("<image>", "").strip()
            prompt += "USER: "
            prompt += value + " "
        else:
            assert i["from"] == "gpt"
            prompt += "ASSISTANT: "
            prompt += value + " "
    return prompt.strip()


def generate_text_data(annotation_path, text_data_save_path):
    if os.path.exists(text_data_save_path):
        print("Text data already exists.")
        return

    with open(annotation_path, "r") as f:
        anno = json.load(f)

    text_data = {}
    for i in tqdm.tqdm(anno):
        prompt = get_text_instruction(i["conversations"])
        unique_idx = i["unique_idx"]

        if "image" in i:
            text_data[unique_idx] = {"image": i["image"], "text": prompt}
        else:
            text_data[unique_idx] = {"text": prompt}

    print("length:", len(text_data), flush=True)

    with open(text_data_save_path, "w") as f:
        json.dump(text_data, f)

    print("Text data generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_annotation_path", type=str, default="./data/llava_instruct_158k.json"
    )
    parser.add_argument(
        "--new_annotation_save_path",
        type=str,
        default="./data/llava_instruct_158k_add_idx.json",
    )
    parser.add_argument(
        "--text_data_save_path", type=str, default="./data/text_data.json"
    )
    args = parser.parse_args()
    add_idx(args.raw_annotation_path, args.new_annotation_save_path)
    generate_text_data(args.new_annotation_save_path, args.text_data_save_path)
