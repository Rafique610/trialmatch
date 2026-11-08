"""Download the published adapter into checkpoints/final/ (no login needed).
Run: uv run python -m scripts.download_adapter [--repo <user>/<repo>]
Then: uv run python -m src.finetune.predict --split val --run finetuned_val --adapter checkpoints/final
"""

import argparse

from huggingface_hub import snapshot_download

from src.config import settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=settings.adapter_repo_id)
    args = ap.parse_args()
    if not args.repo:
        raise SystemExit("No adapter repo set: pass --repo or set TM_ADAPTER_REPO_ID")
    out = settings.checkpoints_dir / "final"
    snapshot_download(args.repo, local_dir=out, allow_patterns=["adapter_*", "README.md"])
    print(f"{args.repo} -> {out}")


if __name__ == "__main__":
    main()
