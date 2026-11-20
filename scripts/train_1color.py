"""Fine-tune SmolVLA on an existing one-candidate colour-matching dataset."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download
from lerobot.scripts import lerobot_train

if __package__:
    from .smolvla_640 import enable_640x480
else:
    from smolvla_640 import enable_640x480


def update_windows_checkpoint(checkpoint: Path) -> None:
    import _winapi

    last = checkpoint.parent.resolve() / "last"
    if last.is_junction():
        last.rmdir()
    elif last.is_symlink():
        last.unlink()
    _winapi.CreateJunction(str(checkpoint.resolve()), str(last))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="LeRobot dataset directory")
    source.add_argument("--resume", type=Path, help="Checkpoint's train_config.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, help="Total steps (default 5000 for a new run)")
    args = parser.parse_args()
    enable_640x480()
    if os.name == "nt":
        lerobot_train.update_last_checkpoint = update_windows_checkpoint

    if args.resume:
        cli = [f"--config_path={args.resume.resolve(strict=True)}", "--resume=true"]
    else:
        root = args.dataset.resolve()
        if not (root / "meta" / "info.json").is_file():
            parser.error(f"Missing dataset metadata in {root}")
        model = snapshot_download(
            "lerobot/smolvla_base",
            revision="5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6",
            allow_patterns=["*.json", "*.safetensors"],
        )
        if args.output is None:
            args.output = Path("outputs/train") / datetime.now().strftime("1color-%Y%m%d-%H%M%S")
        if args.steps is None:
            args.steps = 5000
        cli = [
            "--dataset.repo_id=local/so101_1color",
            f"--dataset.root={root}",
            f"--policy.path={model}",
            "--policy.input_features=null",
            "--policy.load_vlm_weights=false",
            "--policy.train_expert_only=true",
            "--policy.train_state_proj=false",
            "--policy.freeze_vision_encoder=true",
            "--policy.resize_imgs_with_padding=[640,480]",
            "--policy.push_to_hub=false",
            "--policy.device=cuda",
            "--accelerator.mixed_precision=bf16",
            "--batch_size=1",
            "--num_workers=0",
            "--log_freq=10",
            "--save_freq=5000",
            "--env_eval_freq=0",
            "--trackio.enable=false",
        ]
    if args.output is not None:
        cli.append(f"--output_dir={args.output.resolve()}")
    if args.steps is not None:
        if args.steps <= 0:
            parser.error("--steps must be positive")
        cli.append(f"--steps={args.steps}")
    sys.argv = ["lerobot-train", *cli]
    lerobot_train.main()


if __name__ == "__main__":
    main()
