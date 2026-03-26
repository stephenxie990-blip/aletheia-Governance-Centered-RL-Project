#!/usr/bin/env python
"""
Aletheia CartPole 训练脚本（薄壳）
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("ALETHEIA_QUIET", "1")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# =============================================================================
# 参数解析
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ALETHEIA CartPole Training (Unified API)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 环境参数
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Gymnasium environment ID")
    parser.add_argument("--steps", type=int, default=80000,
                        help="Target environment steps (used when --update-steps is not set)")
    parser.add_argument("--update-steps", type=int, default=None,
                        help="Total training update steps (override env steps target)")
    parser.add_argument("--collect-steps-per-cycle", type=int, default=32,
                        help="Environment steps collected per train cycle")
    parser.add_argument("--train-steps-per-cycle", type=int, default=4,
                        help="Gradient updates per train cycle")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cpu/cuda)")
    
    # 模型参数
    parser.add_argument("--save", type=str, default=None,
                        help="Save checkpoint directory")
    parser.add_argument("--load", type=str, default=None,
                        help="Load checkpoint path")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from trainer_state_*.pt or resume_latest.pt")
    parser.add_argument("--overrides", type=str, default=None,
                        help="JSON config overrides (file path or string)")
    
    # 训练参数
    parser.add_argument("--wm-seq-len", type=int, default=16,
                        help="World model sequence length")
    parser.add_argument("--wm-batch-size", type=int, default=64,
                        help="World model batch size")
    parser.add_argument("--imagination-horizon", type=int, default=15,
                        help="Imagination horizon")
    parser.add_argument("--imagination-batch-size", type=int, default=8,
                        help="Imagination batch size")
    parser.add_argument("--rl-batch-size", type=int, default=64,
                        help="RL batch size")
    parser.add_argument("--buffer-capacity", type=int, default=10000,
                        help="Replay buffer capacity (episodes)")
    parser.add_argument("--pretrain-ratio", type=float, default=0.05,
                        help="World model pretrain ratio")
    parser.add_argument("--warmup-ratio", type=float, default=0.0,
                        help="Warmup ratio")
    parser.add_argument("--imagination-only", action=argparse.BooleanOptionalAction, default=False,
                        help="Use pure imagination policy updates (recommended only after stable real-data learning)")
    parser.add_argument("--imag-ratio-start", type=float, default=0.0,
                        help="Initial imagined-batch ratio")
    parser.add_argument("--imag-ratio-end", type=float, default=0.4,
                        help="Final imagined-batch ratio")
    parser.add_argument("--imag-ratio-max", type=float, default=0.6,
                        help="Upper bound for imagined-batch ratio")
    parser.add_argument("--imag-ratio-ramp-steps", type=int, default=2000,
                        help="Ramp steps for imagined-batch ratio after warmup")
    parser.add_argument("--imag-gradient", type=str, default="dynamics",
                        choices=("dynamics", "reinforce", "both"),
                        help="Actor gradient mode on imagined trajectories")
    parser.add_argument("--imag-gradient-mix", type=float, default=0.0,
                        help="Mix coefficient used when imag-gradient=both")
    parser.add_argument("--actor-analytic-weight", type=float, default=1.0,
                        help="Weight for analytic actor gradients on imagined trajectories")
    parser.add_argument("--actor-reinforce-aux-weight-discrete", type=float, default=0.1,
                        help="Auxiliary REINFORCE weight added for discrete imagined policies")
    parser.add_argument("--use-reward-ema", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Dreamer reward EMA normalization in actor targets")
    
    # 日志参数
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Log interval (steps)")
    parser.add_argument("--eval-interval", type=int, default=1000,
                        help="Evaluation interval (steps)")
    parser.add_argument("--save-interval", type=int, default=5000,
                        help="Save interval (steps)")
    parser.add_argument("--eval-episodes", type=int, default=15,
                        help="Evaluation episodes")
    parser.add_argument("--eval-max-steps", type=int, default=500,
                        help="Max steps per evaluation episode")
    
    # 开关
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only")
    parser.add_argument("--enable-eval", action="store_true",
                        help="Enable evaluation during training")
    parser.add_argument("--render", action="store_true",
                        help="Render environment")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")
    
    return parser.parse_args()


def _sanitize_run_label(raw: str) -> str:
    text = (raw or "").strip()
    if text.endswith(".json"):
        text = Path(text).stem
    if text.endswith("_overrides"):
        text = text[: -len("_overrides")]
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "cartpole_train"


def _infer_total_updates(args: Any) -> int:
    if getattr(args, "update_steps", None):
        return max(1, int(args.update_steps))
    collect_steps = max(1, int(getattr(args, "collect_steps_per_cycle", 32)))
    steps = max(1, int(getattr(args, "steps", 80000)))
    return max(1, int(math.ceil(float(steps) / float(collect_steps))))


def _build_default_save_dir(
    args: Any,
    *,
    root: Path | None = None,
    now: datetime | None = None,
) -> Path:
    outputs_root = Path(root) if root is not None else PROJECT_ROOT / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    label_source = getattr(args, "overrides", None) or getattr(args, "env", "cartpole_train")
    label = _sanitize_run_label(str(label_source))
    seed = int(getattr(args, "seed", 42))
    total_updates = _infer_total_updates(args)
    base_name = f"exp_seed{seed}_{label}_{total_updates}_{timestamp}"
    candidate = outputs_root / base_name
    suffix = 1
    while candidate.exists():
        candidate = outputs_root / f"{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


def _apply_default_save_dir(args: Any) -> Path | None:
    if getattr(args, "save", None):
        return None
    if bool(getattr(args, "eval_only", False)):
        return None
    save_dir = _build_default_save_dir(args)
    args.save = str(save_dir)
    return save_dir


# =============================================================================
# 主函数
# =============================================================================

def main() -> int:
    """主函数"""
    args = parse_args()
    auto_save_dir = _apply_default_save_dir(args)
    if auto_save_dir is not None:
        print(f"Auto save dir: {auto_save_dir}", flush=True)
    args.argv = list(sys.argv)
    try:
        from aletheia.aletheia_api import run_train
    except ImportError as e:
        print(f"Failed to import aletheia_api: {e}", flush=True)
        print("Make sure aletheia package is in PYTHONPATH", flush=True)
        return 1

    result = run_train(args)
    return int(result.get("status", 1))


if __name__ == "__main__":
    sys.exit(main())
