import os
import csv
import argparse
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# Initialize tokenizers with trust_remote_code enabled.
tokenizers = {
    "llama_3_1_8b": AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True, use_fast=False
    ),
    "llama_3_1_70b": AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-70B-Instruct", trust_remote_code=True, use_fast=False
    ),
    "llama_3_2_3b": AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct", trust_remote_code=True, use_fast=False
    ),
    "llama_3_3_70b": AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.3-70B-Instruct", trust_remote_code=True, use_fast=False
    ),
    "deepseek_r1_14b": AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", trust_remote_code=True, use_fast=False
    ),
    "deepseek_r1_70b": AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", trust_remote_code=True, use_fast=False
    ),
    "qwen_2_5_7b": AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True, use_fast=False
    ),
    "qwen_2_5_72b": AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-72B-Instruct", trust_remote_code=True, use_fast=False
    ),
    "mistral_7b": AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3", trust_remote_code=True, use_fast=True
    ),
    "mistral_24b": AutoTokenizer.from_pretrained(
        "mistralai/Mistral-Small-24B-Instruct-2501", trust_remote_code=True, use_fast=True
    ),
}

# baseline for NSL comparison
baseline_tok = tokenizers["llama_3_1_8b"]

def token_count_per_row(input_dir):
    # Directories for storing results
    base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    results_dir = os.path.join(base_dir, "tokenization-results")
    count_dir   = os.path.join(results_dir, "count-per-row")
    plot_dir    = os.path.join(results_dir, "plots-per-row")

    os.makedirs(count_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    for name in tokenizers:
        os.makedirs(os.path.join(count_dir, name), exist_ok=True)

    # Data structures: track tokens, bytes/token, and NSL per model per dataset
    efficiency_data = {
        name: {"tokens": {}, "bytes_per_token": {}, "nsl": {}}
        for name in tokenizers
    }
    dataset_names = []

    # Iterate through datasets
    for subdir in sorted(os.listdir(input_dir)):
        subdir_path = os.path.join(input_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        dataset_names.append(subdir)

        # find the first *_responses.csv
        for filename in os.listdir(subdir_path):
            if not filename.endswith("_responses.csv"):
                continue

            tokens_per_model       = {n: [] for n in tokenizers}
            bytes_per_token_model  = {n: [] for n in tokenizers}
            nsl_per_model          = {n: [] for n in tokenizers}

            with open(os.path.join(subdir_path, filename), "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = " ".join([
                        row.get("System Prompt", ""),
                        row.get("Prompt", "")
                    ]).strip()
                    # bytes in UTF-8
                    byte_len = len(text.encode("utf-8"))
                    # baseline token count
                    base_count = len(baseline_tok(text)["input_ids"]) or 1

                    for name, tok in tokenizers.items():
                        ids  = tok(text)["input_ids"]
                        tlen = len(ids)
                        tokens_per_model[name].append(tlen)
                        # Bytes per Token
                        bytes_per_token_model[name].append(byte_len / tlen if tlen else 0)
                        # Normalized Sequence Length
                        nsl_per_model[name].append(tlen / base_count)

            # compute averages for this dataset
            for name in tokenizers:
                ctoks = tokens_per_model[name]
                cbpt  = bytes_per_token_model[name]
                cnsl  = nsl_per_model[name]

                efficiency_data[name]["tokens"][subdir]          = sum(ctoks)/len(ctoks)   if ctoks else 0
                efficiency_data[name]["bytes_per_token"][subdir] = sum(cbpt) /len(cbpt)    if cbpt  else 0
                efficiency_data[name]["nsl"][subdir]             = sum(cnsl) /len(cnsl)    if cnsl  else 0

            break  # only first CSV per dataset

    # write per-model CSVs
    for name, data in efficiency_data.items():
        out_csv = os.path.join(count_dir, name, "average_tokens_per_row.csv")
        with open(out_csv, "w", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "dataset",
                "avg_tokens_per_row",
                "avg_bytes_per_token",
                "avg_normalized_seq_len"
            ])
            for ds in dataset_names:
                w.writerow([
                    ds,
                    data["tokens"].get(ds, 0),
                    data["bytes_per_token"].get(ds, 0),
                    data["nsl"].get(ds, 0),
                ])

    # (optional) Plots
    plt.figure(figsize=(12, 6))
    markers   = ["o","s","^","v","<",">","*","x","D","p"]
    linestyles= ["-","--","-.",":"]
    for idx, name in enumerate(tokenizers):
        y = [efficiency_data[name]["tokens"].get(ds,0) for ds in dataset_names]
        plt.plot(
            dataset_names, y,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            label=name
        )
    plt.grid(True)
    plt.xticks(rotation=45, ha="right")
    plt.title("Tokenizer Efficiency (Avg Tokens Per Row)")
    plt.xlabel("Dataset")
    plt.ylabel("Average Tokens")
    plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "tokenizer_efficiency_per_row.png"))
    plt.close()


def token_count_per_word(input_dir):
    # Directories for storing results
    base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    results_dir = os.path.join(base_dir, "tokenization-results")
    count_dir   = os.path.join(results_dir, "count-per-word")
    plot_dir    = os.path.join(results_dir, "plots-per-word")

    os.makedirs(count_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    for name in tokenizers:
        os.makedirs(os.path.join(count_dir, name), exist_ok=True)

    # Data structures
    efficiency_data = {
        name: {"tokens": {}, "bytes_per_token": {}, "nsl": {}}
        for name in tokenizers
    }
    dataset_names = []

    for subdir in sorted(os.listdir(input_dir)):
        subdir_path = os.path.join(input_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        dataset_names.append(subdir)

        for filename in os.listdir(subdir_path):
            if not filename.endswith("_responses.csv"):
                continue

            tokens_per_model       = {n: [] for n in tokenizers}
            bytes_per_token_model  = {n: [] for n in tokenizers}
            nsl_per_model          = {n: [] for n in tokenizers}

            with open(os.path.join(subdir_path, filename), "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = " ".join([
                        row.get("System Prompt", ""),
                        row.get("Prompt", "")
                    ]).strip()
                    word_count = len(text.split()) or 1
                    byte_len   = len(text.encode("utf-8"))
                    base_count = len(baseline_tok(text)["input_ids"]) or 1

                    for name, tok in tokenizers.items():
                        ids        = tok(text)["input_ids"]
                        tlen       = len(ids)
                        # tokens per word ratio
                        tokens_per_model[name].append(tlen / word_count)
                        # bytes per token
                        bytes_per_token_model[name].append(byte_len / tlen if tlen else 0)
                        # normalized seq len
                        nsl_per_model[name].append(tlen / base_count)

            for name in tokenizers:
                cratios = tokens_per_model[name]
                cbpt    = bytes_per_token_model[name]
                cnsl    = nsl_per_model[name]

                efficiency_data[name]["tokens"][subdir]          = sum(cratios)/len(cratios) if cratios else 0
                efficiency_data[name]["bytes_per_token"][subdir] = sum(cbpt)   /len(cbpt)    if cbpt    else 0
                efficiency_data[name]["nsl"][subdir]             = sum(cnsl)   /len(cnsl)    if cnsl    else 0

            break

    # write per-model CSVs
    for name, data in efficiency_data.items():
        out_csv = os.path.join(count_dir, name, "average_tokens_per_word.csv")
        with open(out_csv, "w", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "dataset",
                "avg_tokens_per_word",
                "avg_bytes_per_token",
                "avg_normalized_seq_len"
            ])
            for ds in dataset_names:
                w.writerow([
                    ds,
                    data["tokens"].get(ds, 0),
                    data["bytes_per_token"].get(ds, 0),
                    data["nsl"].get(ds, 0),
                ])

    # (optional) plot only the tokens-per-word metric
    plt.figure(figsize=(12, 6))
    markers   = ["o","s","^","v","<",">","*","x","D","p"]
    linestyles= ["-","--","-.",":"]
    for idx, name in enumerate(tokenizers):
        y = [efficiency_data[name]["tokens"].get(ds,0) for ds in dataset_names]
        plt.plot(
            dataset_names, y,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            label=name
        )
    plt.grid(True)
    plt.xticks(rotation=45, ha="right")
    plt.title("Tokenizer Efficiency (Avg Tokens per Word)")
    plt.xlabel("Dataset")
    plt.ylabel("Tokens per Word")
    plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "tokenizer_efficiency_per_word.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="../inference-outputs",
        help="Path to folder containing subdirectories with CSV files",
    )
    args = parser.parse_args()
    token_count_per_row(args.input_dir)
    token_count_per_word(args.input_dir)


if __name__ == "__main__":
    main()
