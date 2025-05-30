from pathlib import Path
import pandas as pd
import sys

# Script location
script_dir = Path(__file__).resolve().parent

# mapping
MODEL_RENAMING = {
    "llama_3_1_8b":   "llama-3.1-8B",
    "llama_3_1_70b":  "llama-3.1-70B",
    "llama_3_2_3b":   "llama-3.2-3B",
    "llama_3_3_70b":  "llama-3.3-70B",
    "qwen_2_5_7b":    "qwen2.5-7B",
    "qwen_2_5_72b":   "qwen2.5-72B",
    "mistral_7b":     "mistral-7B",
    "mistral_24b":    "mistral-24B",
    "deepseek_r1_14b":"deepseek-qwen-14B",
    "deepseek_r1_70b":"deepseek-llama-70B",
}


def token_summarizer_per_row():
    base = script_dir.parent / "tokenization-results" / "count-per-row"
    if not base.exists():
        print(f"Could not find directory {base}", file=sys.stderr)
        sys.exit(1)

    all_dfs = []
    for folder in base.iterdir():
        if not folder.is_dir():
            continue
        key = folder.name
        pretty = MODEL_RENAMING.get(key)
        if pretty is None:
            print(f"Skipping unmapped folder {key}", file=sys.stderr)
            continue

        csv_path = folder / "average_tokens_per_row.csv"
        if not csv_path.exists():
            print(f"Missing file {csv_path}", file=sys.stderr)
            continue

        df = pd.read_csv(csv_path)
        # split off language suffix
        df["lang"]    = df["dataset"].str.rsplit("-", n=1).str[-1]
        df["dataset"] = df["dataset"].str.rsplit("-", n=1).str[0]
        df["model"]   = pretty
        all_dfs.append(df)

    if not all_dfs:
        print("No data frames to concatenate!", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined[[
        "model",
        "dataset",
        "lang",
        "avg_tokens_per_row",
        "avg_bytes_per_token",
        "avg_normalized_seq_len",
    ]]
    combined = combined.rename(columns={
        "model":                   "Model",
        "dataset":                 "Dataset",
        "lang":                    "Language",
        "avg_tokens_per_row":      "Average Token Count Per Row",
        "avg_bytes_per_token":     "Average Bytes Per Token",
        "avg_normalized_seq_len":  "Average Normalized Sequence Length",
    })

    out_path = base / "combined_average_tokens_per_row.csv"
    combined.to_csv(out_path, index=False)
    print(f"Combined tokenization efficiency (per row) saved to: {out_path}")


def token_summarizer_per_word():
    base = script_dir.parent / "tokenization-results" / "count-per-word"
    if not base.exists():
        print(f"Could not find directory {base}", file=sys.stderr)
        sys.exit(1)

    all_dfs = []
    for folder in base.iterdir():
        if not folder.is_dir():
            continue
        key = folder.name
        pretty = MODEL_RENAMING.get(key)
        if pretty is None:
            print(f"Skipping unmapped folder {key}", file=sys.stderr)
            continue

        csv_path = folder / "average_tokens_per_word.csv"
        if not csv_path.exists():
            print(f"Missing file {csv_path}", file=sys.stderr)
            continue

        df = pd.read_csv(csv_path)
        # split off language suffix
        df["lang"]    = df["dataset"].str.rsplit("-", n=1).str[-1]
        df["dataset"] = df["dataset"].str.rsplit("-", n=1).str[0]
        df["model"]   = pretty
        all_dfs.append(df)

    if not all_dfs:
        print("No data frames to concatenate for per-word!", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined[[
        "model",
        "dataset",
        "lang",
        "avg_tokens_per_word",
        "avg_bytes_per_token",
        "avg_normalized_seq_len",
    ]]
    combined = combined.rename(columns={
        "model":                   "Model",
        "dataset":                 "Dataset",
        "lang":                    "Language",
        "avg_tokens_per_word":     "Average Token Count Per Word",
        "avg_bytes_per_token":     "Average Bytes Per Token",
        "avg_normalized_seq_len":  "Average Normalized Sequence Length",
    })

    out_path = base / "combined_average_tokens_per_word.csv"
    combined.to_csv(out_path, index=False)
    print(f"Combined tokenization efficiency (per word) saved to: {out_path}")


def __main__():
    token_summarizer_per_row()
    token_summarizer_per_word()


if __name__ == "__main__":
    __main__()
