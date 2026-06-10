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
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = device_auto()
    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer = Tokenizer({tok: i for i, tok in enumerate(ckpt["vocab"])}, ckpt["vocab"])
    cfg = ckpt["config"]
    task = build_task(cfg["task"]["name"])
    model = build_model(cfg, tokenizer).to(device)
    model.load_state_dict(ckpt["model"])
    metrics = evaluate_buckets(
        model,
        tokenizer,
        task,
        args.samples_per_length,
        cfg.get("generation", {}).get("max_new_tokens", 520),
        device,
    )
    if args.out:
        save_json(metrics, args.out)
    print(metrics)


if __name__ == "__main__":
    main()
