"""Dataset builders for training and evaluation splits."""

from __future__ import annotations

from datasets import Dataset


def _split_offset(split: str) -> int:
    if split == "train":
        return 0
    if split == "eval":
        return 1_000_000
    if split == "test":
        return 2_000_000
    raise ValueError(f"unsupported split: {split}")


def build_salesbench_dataset(
    *,
    split: str,
    num_examples: int,
    base_seed: int,
    base_num_leads: int,
    work_days: int,
    hours_per_day: int,
) -> Dataset:
    """Build a deterministic synthetic dataset of episode seeds/configurations."""

    if num_examples <= 0:
        raise ValueError("num_examples must be > 0")

    rows: list[dict] = []
    seed_offset = _split_offset(split)

    for idx in range(num_examples):
        curriculum = idx % 3
        lead_scale = (0.75, 1.00, 1.25)[curriculum]
        scenario_num_leads = max(12, int(base_num_leads * lead_scale))

        episode_seed = base_seed + seed_offset + idx
        difficulty = ("easy", "medium", "hard")[curriculum]

        rows.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            "Run a sales pipeline episode. Use tools to identify qualified "
                            "leads, make compliant calls, and maximize converted monthly premium."
                        ),
                    }
                ],
                "seed": episode_seed,
                "num_leads": scenario_num_leads,
                "work_days": work_days,
                "hours_per_day": hours_per_day,
                "split": split,
                "difficulty": difficulty,
                "info": {
                    "episode_index": idx,
                    "split": split,
                    "difficulty": difficulty,
                    "seed": episode_seed,
                },
            }
        )

    return Dataset.from_list(rows)
