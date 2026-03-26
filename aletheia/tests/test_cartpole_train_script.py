import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts import cartpole_train


class CartpoleTrainScriptTests(unittest.TestCase):
    def test_build_default_save_dir_uses_override_stem_and_update_steps(self):
        args = SimpleNamespace(
            save=None,
            eval_only=False,
            seed=42,
            env="CartPole-v1",
            overrides="tmp/v214_phase4_final_effective_contact_surface_overrides.json",
            update_steps=2500,
            collect_steps_per_cycle=32,
            steps=80000,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = cartpole_train._build_default_save_dir(
                args,
                root=Path(tmpdir),
                now=datetime(2026, 3, 18, 16, 45, 30),
            )
        self.assertEqual(
            save_dir.name,
            "exp_seed42_v214_phase4_final_effective_contact_surface_2500_20260318_164530",
        )

    def test_apply_default_save_dir_skips_eval_only_and_explicit_save(self):
        args_eval_only = SimpleNamespace(
            save=None,
            eval_only=True,
            seed=7,
            env="CartPole-v1",
            overrides=None,
            update_steps=10,
            collect_steps_per_cycle=32,
            steps=320,
        )
        self.assertIsNone(cartpole_train._apply_default_save_dir(args_eval_only))
        self.assertIsNone(args_eval_only.save)

        args_with_save = SimpleNamespace(
            save="outputs/manual_dir",
            eval_only=False,
            seed=7,
            env="CartPole-v1",
            overrides=None,
            update_steps=10,
            collect_steps_per_cycle=32,
            steps=320,
        )
        self.assertIsNone(cartpole_train._apply_default_save_dir(args_with_save))
        self.assertEqual(args_with_save.save, "outputs/manual_dir")

    def test_build_default_save_dir_avoids_colliding_directory_name(self):
        args = SimpleNamespace(
            save=None,
            eval_only=False,
            seed=42,
            env="CartPole-v1",
            overrides="tmp/v214_phase4_final_effective_contact_surface_overrides.json",
            update_steps=2500,
            collect_steps_per_cycle=32,
            steps=80000,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            existing = root / "exp_seed42_v214_phase4_final_effective_contact_surface_2500_20260318_164530"
            existing.mkdir()
            save_dir = cartpole_train._build_default_save_dir(
                args,
                root=root,
                now=datetime(2026, 3, 18, 16, 45, 30),
            )
        self.assertEqual(
            save_dir.name,
            "exp_seed42_v214_phase4_final_effective_contact_surface_2500_20260318_164530_01",
        )


if __name__ == "__main__":
    unittest.main()
