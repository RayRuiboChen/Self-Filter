import os
import json
import tqdm
import sys
import argparse


def get_gpt_score(response_dir, save_path):
    """
    Parse the GPT outputs
    """
    result_dict = {}
    cnt_dict = {}

    for filename in tqdm.tqdm(os.listdir(response_dir), file=sys.stdout):
        with open(os.path.join(response_dir, filename), "r") as f:
            cur_dict = json.load(f)

        unique_idx = filename.split(".")[0]
        output = cur_dict["choices"][0]["message"]["content"]
        score_line = output.strip().split("\n")[0].split("/")[0].split(" ")[-1]

        try:
            cur_score = float(score_line)
        except:
            cur_score = None
        result_dict[unique_idx] = cur_score

        if cur_score not in cnt_dict:
            cnt_dict[cur_score] = 0
        cnt_dict[cur_score] += 1

    with open(save_path, "w") as f:
        json.dump(result_dict, f)
    print("Raw score extracted.")


def norm_scores(score_dict: dict):
    min_score = min(score_dict.values())
    max_score = max(score_dict.values())

    normed_score_dict = {
        i[0]: (i[1] - min_score) / (max_score - min_score) * 2 - 1
        for i in score_dict.items()
    }

    return normed_score_dict


def process_none_scores(data_dir, raw_name, output_name):
    """
    replace None with avg score
    """
    with open(os.path.join(data_dir, raw_name), "r") as f:
        raw_scores = json.load(f)

    tot_score = 0
    tot_valid_num = 0

    for unique_idx in raw_scores:
        cur_score = raw_scores[unique_idx]
        if cur_score != None:
            tot_score += cur_score
            tot_valid_num += 1

    avg_score = tot_score / tot_valid_num

    # print(avg_score)
    # print(tot_valid_num)

    new_result_dict = {}

    for unique_idx in raw_scores:
        cur_score = raw_scores[unique_idx]
        if cur_score != None:
            new_result_dict[unique_idx] = cur_score
        else:
            new_result_dict[unique_idx] = avg_score

    with open(os.path.join(data_dir, output_name), "w") as f:
        json.dump(new_result_dict, f)

    print("Processed score extracted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--score_dir", type=str, default="./data/scores")
    parser.add_argument("--response_dir", type=str, default="./data/gpt_responses")

    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--raw_score_filename", type=str, default="raw_score.json")
    parser.add_argument(
        "--processed_score_filename", type=str, default="processed_score.json"
    )
    args = parser.parse_args()

    model_response_dir = os.path.join(args.response_dir, args.model_name)
    model_score_dir = os.path.join(args.score_dir, args.model_name)

    os.makedirs(model_score_dir, exist_ok=True)

    raw_score_path = os.path.join(model_score_dir, args.raw_score_filename)
    processed_score_path = os.path.join(model_score_dir, args.processed_score_filename)

    get_gpt_score(model_response_dir, save_path=raw_score_path)
    process_none_scores(
        model_score_dir, args.raw_score_filename, args.processed_score_filename
    )
