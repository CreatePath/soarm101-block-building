"""Check the saved LeRobot rows and the one-candidate collection quotas."""

import argparse
import json
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

if __package__:
    from .collect_1color import BLOCKS, GRID, LEFT, RIGHT, TARGET_EPISODES
    from .collect_pilot import TASK
else:
    from collect_1color import BLOCKS, GRID, LEFT, RIGHT, TARGET_EPISODES
    from collect_pilot import TASK


def check_run(run, expected=TARGET_EPISODES):
    root = run / "dataset"
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    assert expected in (240, TARGET_EPISODES)
    assert len(episodes) == info["total_episodes"] == expected and info["fps"] == 30
    batches = expected // 240
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    assert len(tasks) == 1 and tasks[0]["task_index"] == 0 and tasks[0]["task"] == TASK
    ranges = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        ranges.extend(pq.read_table(path, columns=["episode_index", "length", "dataset_from_index",
                                                    "dataset_to_index"]).to_pylist())
    assert len(ranges) == len(episodes)
    pairs = Counter((e["reference"], e["candidate"]) for e in episodes)
    assert all(pairs[a, b] == batches * (48 if a == b else 4) for a in BLOCKS for b in BLOCKS)
    assert all(sum(e["match"] for e in episodes[i:i+5]) == 4 for i in range(0, expected, 5))
    for side, key, bounds in [("left", "reference_xy", LEFT), ("right", "candidate_xy", RIGHT)]:
        counts = Counter()
        for e in episodes:
            uv = (np.asarray(e[key]) - bounds[0]) / (bounds[1] - bounds[0])
            assert (uv >= 0).all() and (uv < 1).all()
            col, row = np.floor(uv * GRID).astype(int)
            cell = row * GRID[0] + col
            assert cell == e[side + "_cell"]
            counts[e["reference"], e["match"], cell] += 1
        assert all(counts[c, m, k] == batches * (4 if m else 1)
                   for c in BLOCKS for m in (True, False) for k in range(12))

    rows_read = i = 0
    home = None
    for path in sorted((root / "data").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            t = parquet.read_row_group(group)
            e = episodes[i]
            n = t.num_rows
            assert n == e["frames"] and e["index"] == i
            assert ranges[i] == dict(episode_index=i, length=n, dataset_from_index=rows_read,
                                     dataset_to_index=rows_read+n)
            assert np.all(t["episode_index"].to_numpy() == i)
            assert np.array_equal(t["index"].to_numpy(), rows_read + np.arange(n))
            assert np.array_equal(t["frame_index"].to_numpy(), np.arange(n))
            assert np.allclose(t["timestamp"].to_numpy(), np.arange(n)/30, atol=1e-6)
            assert np.all(t["task_index"].to_numpy() == 0)
            state = np.asarray(t["observation.state"].to_pylist())
            action = np.asarray(t["action"].to_pylist())
            assert state.shape == action.shape == (n, 6)
            assert np.isfinite(state).all() and np.isfinite(action).all()
            assert min(e["visible_pixels"]) >= 80 and max(e["ik_error_m"]) <= .003
            assert e["match"] == (e["reference"] == e["candidate"])
            assert e["home_error_rad"] < .01 and e["final_command_is_home"]
            assert np.max(np.abs(state[-1] - state[0])) < .01
            if "home_command_rad" in e:
                target = np.asarray(e["home_command_rad"], dtype=np.float32)
                assert np.array_equal(action[-1], target)
                assert np.max(np.abs(state[0] - target)) < .03
                assert np.linalg.norm(np.array(e["start_xyz_m"][:2]) - e["candidate_xy"]) < .001
            if home is None:
                home = action[-1]
            assert np.array_equal(action[-1], home)
            if e["match"]:
                assert e["pinched_in_air"] and e["lift_m"] > .03 and abs(e["final_xyz_m"][0] - .3875) <= .003
                assert abs(e["final_xyz_m"][2] - e["start_xyz_m"][2]) < .001
                assert np.max(np.abs(state - state[0])) > .1
            else:
                assert np.all(action == home) and np.max(np.abs(state - state[0])) < 1e-4
                assert e["max_block_motion_m"] < 1e-5
            for camera in ("top", "gripper"):
                frames = t["observation.images." + camera]
                for j, frame in enumerate(frames):
                    with Image.open(BytesIO(frame["bytes"].as_py())) as im:
                        assert im.size == (640,480) and im.mode == "RGB"
                        if j in (0, n//2, n-1):
                            im.load()
                        else:
                            im.verify()
            rows_read += n
            i += 1
    assert rows_read == info["total_frames"] and i == expected
    positives = [e for e in episodes if e["match"]]
    summary = dict(episodes=i, frames=rows_read, fps=info["fps"], verified_images=rows_read*2,
                   home_command_deg=np.rad2deg(home.astype(np.float64)).round(2).tolist(),
                   pair_counts={a: {b: pairs[a,b] for b in BLOCKS} for a in BLOCKS},
                   episodes_per_spatial_cell_per_side=20 * batches,
                   max_positive_line_error_mm=1000*max(e["line_error_m"] for e in positives),
                   max_home_error_rad=max(e["home_error_rad"] for e in positives),
                   min_lift_cm=100*min(e["lift_m"] for e in positives))
    (run / "verification.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--episodes", type=int, choices=(240, TARGET_EPISODES), default=TARGET_EPISODES)
    args = parser.parse_args()
    print(json.dumps(check_run(args.run, args.episodes), indent=2))
