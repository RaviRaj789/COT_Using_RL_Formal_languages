import argparse

import torch

from .evaluate import evaluate_buckets
from .tasks import build_task
from .tokenizer import Tokenizer
from .train import build_model
from .utils import device_auto, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples-per-length", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = device_auto()

    ckpt = torch.load(args.checkpoint, map_location=device)

    tokenizer = Tokenizer(
        {tok: i for i, tok in enumerate(ckpt["vocab"])},
        ckpt["vocab"],
    )

    cfg = ckpt["config"]
    task = build_task(cfg["task"]["name"])

    model = build_model(cfg, tokenizer).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Priority:
    # 1. command-line override: --max-new-tokens
    # 2. eval_generation.max_new_tokens from YAML/checkpoint config
    # 3. generation.max_new_tokens from YAML/checkpoint config
    # 4. fallback 520
    if args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens
    else:
        max_new_tokens = cfg.get("eval_generation", {}).get(
            "max_new_tokens",
            cfg.get("generation", {}).get("max_new_tokens", 520),
        )

    metrics = evaluate_buckets(
        model,
        tokenizer,
        task,
        args.samples_per_length,
        max_new_tokens,
        device,
    )

    result = {
        "checkpoint": args.checkpoint,
        "samples_per_length": args.samples_per_length,
        "max_new_tokens": max_new_tokens,
        "metrics": metrics,
    }

    if args.out:
        save_json(result, args.out)

    print(result)


if __name__ == "__main__":
    main()