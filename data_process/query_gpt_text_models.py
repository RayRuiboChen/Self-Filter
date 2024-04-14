import os
import tqdm
import json
import openai
import argparse
import multiprocessing


def warp_inst(text, dimension="quality"):
    inst = f"""We would like to request your feedback on the performance of AI assistant in response to the user's questions in the conversation displayed following.

Conversation: {text}

Please rate according to the {dimension} of the responses to the questions. The assistant should receive a score on a scale of 0 to 10, where a higher score indicates higher level of the {dimension}. Please first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."""
    return inst


def query_gpt(unique_idx):
    unique_idx = str(unique_idx)
    response_save_path = os.path.join(
        args.response_dir, args.gpt_model_name, unique_idx + ".json"
    )
    success = 1

    if os.path.exists(response_save_path):
        return success

    text = text_data[unique_idx]["text"]
    prompt = warp_inst(text)
    try:
        if "instruct" in args.gpt_model_name:
            response = openai.Completion.create(
                model=args.gpt_model_name,
                prompt=prompt,
                seed=args.seed,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        else:
            response = openai.ChatCompletion.create(
                model=args.gpt_model_name,
                messages=[{"role": "user", "content": prompt}],
                seed=args.seed,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        with open(response_save_path, "w") as f:
            json.dump(response, f)
    except Exception as e:
        print("-" * 80, flush=True)
        print(f"Caught error querying with unique idx: {unique_idx}", flush=True)
        print("Error info: ", e, flush=True)
        success = 0

    return success


def main():
    global args
    global text_data

    try:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    except Exception as e:
        print("Error info: ", e, flush=True)
        print("Please first set your OpenAI API Key.", flush=True)
        exit(0)

    parser = argparse.ArgumentParser()

    # [start,end)
    parser.add_argument("--start", type=int, help="start unique index")
    parser.add_argument("--end", type=int, help="end unique idx")
    parser.add_argument("--response_dir", type=str, default="./data/gpt_responses")
    parser.add_argument("--image_dir", type=str, default="./data/coco/train2017")
    parser.add_argument("--text_data_path", type=str, default="./data/text_data.json")

    parser.add_argument("--gpt_model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=10)

    parser.add_argument("--pool_size", type=int, default=8)

    args = parser.parse_args()

    os.makedirs(os.path.join(args.response_dir, args.gpt_model_name), exist_ok=True)

    with open(args.text_data_path, "r") as f:
        text_data = json.load(f)

    if args.end == -1:
        args.end = len(text_data)

    print(f"Evaluating from index {args.start} to {args.end} ...")

    pool = multiprocessing.Pool(processes=args.pool_size)

    tasks = range(args.start, args.end)
    results_generator = pool.imap_unordered(query_gpt, tasks)

    tot_cnt = 0
    tot_success = 0
    for res in results_generator:
        tot_cnt += 1
        tot_success += res

    pool.close()
    pool.join()

    print("-" * 80, flush=True)
    print("Evaluation finished.", flush=True)
    print(f"Total task number: {tot_cnt}", flush=True)
    print(f"Total success number: {tot_success}", flush=True)


if __name__ == "__main__":
    main()
