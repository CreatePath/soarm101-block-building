"""Run one SmolVLA optimizer step on the colour-matching board demonstrations."""

import os
import sys
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download

if __package__:
    from .board_env import IMAGE_HEIGHT, IMAGE_WIDTH
    from .collect_pilot import DATASET_ID, record_episodes
else:
    from board_env import IMAGE_HEIGHT, IMAGE_WIDTH
    from collect_pilot import DATASET_ID, record_episodes


def main() -> None:
    from lerobot.scripts import lerobot_train
    if __package__:
        from .smolvla_640 import enable_640x480
    else:
        from smolvla_640 import enable_640x480

    enable_640x480()
    run = Path("outputs/pilot") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    dataset_root = run.resolve() / "dataset"
    record_episodes(dataset_root)
    model_path = snapshot_download(
        "lerobot/smolvla_base",
        revision="5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6",
        allow_patterns=["*.json", "*.safetensors"],
    )

    if os.name == "nt":
        # Upstream's "last" symlink needs a Windows privilege. Directory junctions
        # provide the same lookup for this single-checkpoint local run.
        import _winapi

        def link_checkpoint(checkpoint: Path) -> None:
            _winapi.CreateJunction(str(checkpoint.resolve()), str(checkpoint.parent.resolve() / "last"))

        lerobot_train.update_last_checkpoint = link_checkpoint

    sys.argv = [
        "lerobot-train",
        f"--dataset.repo_id={DATASET_ID}",
        f"--dataset.root={dataset_root}",
        f"--policy.path={model_path}",
        "--policy.input_features=null",
        "--policy.load_vlm_weights=false",
        "--policy.train_expert_only=true",
        "--policy.train_state_proj=false",
        "--policy.freeze_vision_encoder=true",
        "--policy.push_to_hub=false",
        "--policy.chunk_size=4",
        "--policy.n_action_steps=4",
        f"--policy.resize_imgs_with_padding=[{IMAGE_WIDTH},{IMAGE_HEIGHT}]",
        "--policy.device=cuda",
        "--batch_size=1",
        "--steps=1",
        "--num_workers=0",
        "--log_freq=1",
        "--save_freq=1",
        "--env_eval_freq=0",
        "--trackio.enable=false",
        f"--output_dir={run / 'train'}",
    ]
    lerobot_train.main()


if __name__ == "__main__":
    main()
