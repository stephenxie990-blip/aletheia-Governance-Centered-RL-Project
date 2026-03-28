"""
Aletheia v5.2.5 核心组件单元测试

覆盖:
- TEST-1: StateTransition (RSSM)  initial_state, observe_step, imagine_step, observe_sequence
- TEST-2: Actor  离散/连续策略采样, log_prob, entropy
- TEST-3: Critic  SingleCritic, MultiHeadEnsembleCritic 输出与损失
- TEST-4: compute_lambda_returns + symlog/symexp 一致性
"""
import os
import sys
import unittest
import warnings
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import numpy as np

# 确保仓库根目录在 sys.path
_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from aletheia.aletheia_foundation import (
    RSSMConfig,
    DistributionConfig,
    KLConfig,
    ActorConfig,
    CriticConfig,
    EntropyProtectionConfig,
    DiscreteDistConfig,
    ContinuousDistConfig,
    MultiHorizonContinueConfig,
    compute_lambda_returns,
    symlog,
    symexp,
)
from aletheia.aletheia_config import (
    AuxTasksConfig,
    ConsistencyAuditorConfig,
    ConsistencyHeadConfig,
    ControlConfig,
    ContinueConfig,
    GradientWallConfig,
    LossWeightsConfig,
    MHCConfig,
    MSCConfig,
    NSTConfig,
    ProjectionConfig,
    SCConfig,
    WorldModelConfig,
)
from aletheia.aletheia_world_model import (
    ConsistencyAuditor,
    MultiHorizonContinueHead,
    MultiScaleContinue,
    NStepTerminationLoss,
    PolicyFeatures,
    PredictionHeads,
    PredictionHeadsConfig,
    ShortcutConsistencyLoss,
    StateTransition,
    TransitionOutput,
    WMTrajectory,
    WorldModel,
    _apply_wall_strength_to_policy_features,
)
from aletheia.aletheia_actor_critic import (
    Actor,
    HierarchicalUnifiedCritic,
    SingleCritic,
    MultiHeadEnsembleCritic,
    CriticOutput,
    CategoricalStraightThrough,
    TanhTransformedDistribution,
    # 确认 symlog/symexp 是统一别名
    symlog as ac_symlog,
    symexp as ac_symexp,
)


# =========================================================================
# TEST-1: StateTransition (RSSM)
# =========================================================================

class TestStateTransition(unittest.TestCase):
    """测试 RSSM 状态转移模块核心功能"""

    @classmethod
    def setUpClass(cls):
        cls.B = 4  # batch size
        cls.T = 5  # sequence length
        cls.N = 8  # num_distributions
        cls.K = 8  # num_classes

        dist_cfg = DistributionConfig(
            num_distributions=cls.N,
            num_classes=cls.K,
            use_unimix=False,
        )
        kl_cfg = KLConfig(free_nats=0.0)

        cls.cfg = RSSMConfig(
            deter_dim=64,
            hidden_dim=64,
            action_embed_dim=16,
            obs_embed_dim=32,
            z_embed_dim=32,
            distribution=dist_cfg,
            kl=kl_cfg,
            use_feature_norm=True,
            use_separate_norm=False,
            use_persist_gate=False,
        )
        cls.model = StateTransition(cls.cfg)
        cls.model.eval()

    def test_initial_state_shapes(self):
        """initial_state 应返回正确形状的 (h, z_raw)"""
        h, z_raw = self.model.initial_state(self.B, device=torch.device("cpu"))
        self.assertEqual(h.shape, (self.B, self.cfg.deter_dim))
        self.assertEqual(z_raw.shape, (self.B, self.N * self.K))

    def test_initial_state_with_noise(self):
        """加噪的 initial_state 不应全零"""
        h, z_raw = self.model.initial_state(
            self.B, device=torch.device("cpu"), add_noise=True, noise_scale=1.0
        )
        # h 应该有一些非零分量noise_scale=1.0 时几乎不可能全零
        self.assertTrue(h.abs().sum() > 0)

    def test_observe_step_output_types_and_shapes(self):
        """observe_step 应返回 TransitionOutput 且各字段形状正确"""
        h, z_raw = self.model.initial_state(self.B, device=torch.device("cpu"))
        action_embed = torch.randn(self.B, self.cfg.action_embed_dim)
        obs_embed = torch.randn(self.B, self.cfg.obs_embed_dim)

        out = self.model.observe_step(
            prev_state=(h, z_raw),
            action_embed=action_embed,
            obs_embed=obs_embed,
        )

        self.assertIsInstance(out, TransitionOutput)
        self.assertEqual(out.h.shape, (self.B, self.cfg.deter_dim))
        self.assertEqual(out.z_post_raw.shape, (self.B, self.N * self.K))
        self.assertEqual(out.z_prior_raw.shape, (self.B, self.N * self.K))
        self.assertEqual(out.features_post.shape, (self.B, self.cfg.feat_dim))
        self.assertEqual(out.features_prior.shape, (self.B, self.cfg.feat_dim))
        self.assertEqual(out.kl_raw.shape, (self.B,))
        self.assertTrue(torch.isfinite(out.kl_raw).all())

    def test_imagine_step_output_shapes(self):
        """imagine_step 应只用先验无后验kl_raw 全零"""
        h, z_raw = self.model.initial_state(self.B, device=torch.device("cpu"))
        action_embed = torch.randn(self.B, self.cfg.action_embed_dim)

        out = self.model.imagine_step(
            prev_state=(h, z_raw),
            action_embed=action_embed,
        )

        self.assertEqual(out.h.shape, (self.B, self.cfg.deter_dim))
        # imagine 时 z_post_raw 和 z_prior_raw 应相同
        self.assertTrue(torch.equal(out.z_post_raw, out.z_prior_raw))
        # imagine 时 features_post 和 features_prior 应相同
        self.assertTrue(torch.equal(out.features_post, out.features_prior))
        # kl 应全为零
        self.assertTrue(torch.allclose(out.kl_raw, torch.zeros_like(out.kl_raw)))

    def test_observe_step_kl_positive(self):
        """不同的 obs_embed 应导致 post  priorKL > 0"""
        h, z_raw = self.model.initial_state(self.B, device=torch.device("cpu"))
        action_embed = torch.zeros(self.B, self.cfg.action_embed_dim)
        # 使用较大的 obs_embed 来确保后验和先验有差异
        obs_embed = torch.randn(self.B, self.cfg.obs_embed_dim) * 5.0

        out = self.model.observe_step(
            prev_state=(h, z_raw),
            action_embed=action_embed,
            obs_embed=obs_embed,
        )
        # KL >= 0 是必须的
        self.assertTrue((out.kl_raw >= -1e-6).all().item())

    def test_observe_sequence_output_shapes(self):
        """observe_sequence 应返回正确形状的序列输出"""
        obs_embeds = torch.randn(self.B, self.T, self.cfg.obs_embed_dim)
        action_embeds = torch.randn(self.B, self.T, self.cfg.action_embed_dim)

        result = self.model.observe_sequence(
            obs_embeds=obs_embeds,
            action_embeds=action_embeds,
        )

        # 验证关键输出存在
        self.assertIn("feats_post", result)
        self.assertEqual(result["feats_post"].shape, (self.B, self.T, self.cfg.feat_dim))
        self.assertIn("z_post_raw_seq", result)
        self.assertEqual(result["z_post_raw_seq"].shape, (self.B, self.T, self.N * self.K))

    def test_state_for_next_chaining(self):
        """state_for_next 应可直接用于下一步输入"""
        h, z_raw = self.model.initial_state(self.B, device=torch.device("cpu"))
        action_embed = torch.randn(self.B, self.cfg.action_embed_dim)
        obs_embed = torch.randn(self.B, self.cfg.obs_embed_dim)

        out1 = self.model.observe_step(
            prev_state=(h, z_raw),
            action_embed=action_embed,
            obs_embed=obs_embed,
        )
        # 用 state_for_next 继续
        out2 = self.model.observe_step(
            prev_state=out1.state_for_next,
            action_embed=action_embed,
            obs_embed=obs_embed,
        )

        self.assertIsInstance(out2, TransitionOutput)
        self.assertEqual(out2.h.shape, (self.B, self.cfg.deter_dim))

    def test_get_features(self):
        """get_features 应将 h 和 z_raw 组合为 feat_dim 维特征"""
        h, z_raw = self.model.initial_state(self.B, device=torch.device("cpu"))
        feat = self.model.get_features(h, z_raw)
        self.assertEqual(feat.shape, (self.B, self.cfg.feat_dim))
        self.assertTrue(torch.isfinite(feat).all())


# =========================================================================
# TEST-2: Actor
# =========================================================================

class TestActorDiscrete(unittest.TestCase):
    """测试离散动作 Actor"""

    @classmethod
    def setUpClass(cls):
        cls.feat_dim = 96  # deter_dim + z_embed_dim
        cls.action_dim = 5
        cls.B = 4

        config = ActorConfig(
            hidden_dims=(64, 64),
            entropy_protection=EntropyProtectionConfig(enabled=True),
        )
        cls.actor = Actor(
            feat_dim=cls.feat_dim,
            action_dim=cls.action_dim,
            is_discrete=True,
            config=config,
        )
        cls.actor.eval()

    def test_forward_returns_categorical(self):
        """离散 Actor.forward 应返回 CategoricalStraightThrough"""
        feat = torch.randn(self.B, self.feat_dim)
        dist = self.actor(feat, temperature=1.0)
        self.assertIsInstance(dist, CategoricalStraightThrough)

    def test_sample_shape(self):
        """采样动作应为 (B, action_dim) 的 one-hot 向量"""
        feat = torch.randn(self.B, self.feat_dim)
        dist = self.actor(feat, temperature=1.0)
        action = dist.sample()
        self.assertEqual(action.shape, (self.B, self.action_dim))
        # one-hot: 每行总和约等于 1STE 下是精确 1
        self.assertTrue(torch.allclose(action.sum(dim=-1), torch.ones(self.B), atol=1e-5))

    def test_log_prob_shape(self):
        """log_prob 应返回 (B,) 形状"""
        feat = torch.randn(self.B, self.feat_dim)
        dist = self.actor(feat, temperature=1.0)
        action = dist.sample()
        lp = dist.log_prob(action)
        self.assertEqual(lp.shape, (self.B,))
        self.assertTrue(torch.isfinite(lp).all())

    def test_entropy_shape_and_positive(self):
        """entropy 应为 (B,)且非负"""
        feat = torch.randn(self.B, self.feat_dim)
        dist = self.actor(feat, temperature=1.0)
        ent = dist.entropy()
        self.assertEqual(ent.shape, (self.B,))
        self.assertTrue((ent >= 0).all().item())

    def test_get_action_no_grad(self):
        """get_action 应在 no_grad 下返回动作和信息"""
        feat = torch.randn(self.B, self.feat_dim)
        action, info = self.actor.get_action(feat, temperature=1.0)
        self.assertEqual(action.shape, (self.B, self.action_dim))
        self.assertIn("log_prob", info)
        self.assertIn("entropy", info)

    def test_mode_deterministic(self):
        """mode 属性应返回确定性动作one-hot at argmax"""
        feat = torch.randn(self.B, self.feat_dim)
        dist = self.actor(feat, temperature=1.0)
        mode = dist.mode
        self.assertEqual(mode.shape, (self.B, self.action_dim))
        # mode 应是 one-hot
        self.assertTrue(torch.allclose(mode.sum(dim=-1), torch.ones(self.B), atol=1e-5))


class TestActorContinuous(unittest.TestCase):
    """测试连续动作 Actor"""

    @classmethod
    def setUpClass(cls):
        cls.feat_dim = 96
        cls.action_dim = 3
        cls.B = 4

        config = ActorConfig(
            hidden_dims=(64, 64),
            entropy_protection=EntropyProtectionConfig(enabled=False),
        )
        cls.actor = Actor(
            feat_dim=cls.feat_dim,
            action_dim=cls.action_dim,
            is_discrete=False,
            config=config,
        )
        cls.actor.eval()

    def test_forward_returns_tanh_distribution(self):
        """连续 Actor.forward 应返回 TanhTransformedDistribution"""
        feat = torch.randn(self.B, self.feat_dim)
        dist = self.actor(feat, temperature=1.0)
        self.assertIsInstance(dist, TanhTransformedDistribution)

    def test_sample_bounded(self):
        """连续动作采样应在 [-1, 1] 范围内"""
        feat = torch.randn(self.B, self.feat_dim)
        dist = self.actor(feat, temperature=1.0)
        action = dist.sample()
        self.assertEqual(action.shape, (self.B, self.action_dim))
        self.assertTrue((action >= -1.0 - 1e-3).all().item())
        self.assertTrue((action <= 1.0 + 1e-3).all().item())

    def test_log_prob_shape(self):
        """连续 log_prob 应为 (B,)"""
        feat = torch.randn(self.B, self.feat_dim)
        dist = self.actor(feat, temperature=1.0)
        action = dist.sample()
        lp = dist.log_prob(action)
        self.assertEqual(lp.shape, (self.B,))
        self.assertTrue(torch.isfinite(lp).all())


# =========================================================================
# TEST-3: Critic
# =========================================================================

class TestSingleCritic(unittest.TestCase):
    """测试单头 Critic"""

    @classmethod
    def setUpClass(cls):
        cls.feat_dim = 96
        cls.B = 4

        cls.critic = SingleCritic(
            feat_dim=cls.feat_dim,
            hidden_dim=64,
            depth=2,
            primary_gamma=0.99,
        )
        cls.critic.eval()

    def test_forward_output_type_and_shapes(self):
        """SingleCritic.forward 应返回 CriticOutput 且形状正确"""
        feat = torch.randn(self.B, self.feat_dim)
        out = self.critic(feat)

        self.assertIsInstance(out, CriticOutput)
        self.assertEqual(out.values_symlog_main.shape, (self.B,))
        self.assertEqual(out.values_real_main.shape, (self.B,))
        self.assertEqual(out.uncertainty_raw.shape, (self.B,))

    def test_symlog_clip_applied(self):
        """symlog 值应被 clamp 在 [-clip, clip] 范围内"""
        feat = torch.randn(self.B, self.feat_dim) * 100  # 极端输入
        out = self.critic(feat)
        self.assertTrue((out.values_symlog_main.abs() <= self.critic.symlog_clip + 1e-5).all())

    def test_compute_loss_returns_dict(self):
        """compute_loss 应返回 {'loss': tensor, 'per_gamma': dict}"""
        feat = torch.randn(self.B, self.feat_dim)
        targets = torch.randn(self.B)
        losses = self.critic.compute_loss(feat, targets)
        self.assertIn("loss", losses)
        self.assertIn("per_gamma", losses)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_values_per_gamma_populated(self):
        """values_per_gamma 应包含 primary_gamma 键"""
        feat = torch.randn(self.B, self.feat_dim)
        out = self.critic(feat)
        self.assertIsNotNone(out.values_per_gamma)
        self.assertIn(0.99, out.values_per_gamma)

    def test_critic_output_detach(self):
        """CriticOutput.detach() 应切断所有梯度"""
        feat = torch.randn(self.B, self.feat_dim, requires_grad=True)
        out = self.critic(feat)
        out_d = out.detach()
        self.assertFalse(out_d.values_symlog_main.requires_grad)
        self.assertFalse(out_d.values_real_main.requires_grad)


class TestMultiHeadEnsembleCritic(unittest.TestCase):
    """测试多头集成 Critic"""

    @classmethod
    def setUpClass(cls):
        cls.feat_dim = 96
        cls.B = 4

        cls.cfg = CriticConfig(
            d_feature=cls.feat_dim,
            gammas=(0.9, 0.99),
            n_ensemble=2,
            hidden_dim=64,
            hidden_depth=2,
        )
        cls.critic = MultiHeadEnsembleCritic(cls.cfg)
        cls.critic.eval()

    def test_forward_output_shapes(self):
        """MultiHeadEnsembleCritic.forward 应返回正确形状"""
        feat = torch.randn(self.B, self.feat_dim)
        out = self.critic(feat)

        self.assertIsInstance(out, CriticOutput)
        self.assertEqual(out.values_symlog_main.shape, (self.B,))
        self.assertEqual(out.values_real_main.shape, (self.B,))

    def test_values_per_gamma_all_gammas(self):
        """values_per_gamma 应包含所有配置的 gamma"""
        feat = torch.randn(self.B, self.feat_dim)
        out = self.critic(feat)

        self.assertIsNotNone(out.values_per_gamma)
        for g in self.cfg.gammas:
            self.assertIn(g, out.values_per_gamma)
            self.assertEqual(out.values_per_gamma[g].shape, (self.B,))

    def test_compute_loss_multi_gamma(self):
        """compute_loss 应支持多 gamma 目标"""
        feat = torch.randn(self.B, self.feat_dim)
        targets = {g: torch.randn(self.B) for g in self.cfg.gammas}
        losses = self.critic.compute_loss(feat, targets)
        self.assertIn("loss", losses)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_compute_loss_rejects_missing_gamma_targets(self):
        feat = torch.randn(self.B, self.feat_dim)
        targets = {0.9: torch.randn(self.B)}

        with self.assertRaisesRegex(
            ValueError,
            "gamma targets must exactly match configured critic gammas",
        ):
            self.critic.compute_loss(feat, targets)

    def test_compute_loss_rejects_extra_gamma_targets(self):
        feat = torch.randn(self.B, self.feat_dim)
        targets = {
            0.9: torch.randn(self.B),
            0.95: torch.randn(self.B),
            0.99: torch.randn(self.B),
        }

        with self.assertRaisesRegex(
            ValueError,
            "gamma targets must exactly match configured critic gammas",
        ):
            self.critic.compute_loss(feat, targets)

    def test_hierarchical_critic_rejects_gamma_target_mismatch_with_router_enabled(self):
        critic = HierarchicalUnifiedCritic(
            CriticConfig(
                d_feature=self.feat_dim,
                gammas=(0.9, 0.99),
                n_ensemble=2,
                hidden_dim=64,
                hidden_depth=2,
                mode="adaptive",
                use_adaptive_routing=True,
            )
        )
        feat = torch.randn(self.B, self.feat_dim)
        targets = {0.9: torch.randn(self.B)}

        with self.assertRaisesRegex(
            ValueError,
            "gamma targets must exactly match configured critic gammas",
        ):
            critic.compute_loss(feat, targets)


# =========================================================================
# TEST-4.5: ConsistencyAuditor remaining_steps 计算边界
# =========================================================================

class TestConsistencyAuditorRemainingSteps(unittest.TestCase):
    """测试 remaining_steps 只在真正需要的模块上计算"""

    def _make_trajectory(self, batch_size: int = 2, seq_len: int = 4) -> WMTrajectory:
        vitals_dim = 3
        action_dim = 2
        deter_dim = 5
        stoch_dim = 4
        feat_dim = 6
        return WMTrajectory(
            vitals=torch.zeros(batch_size, seq_len + 1, vitals_dim),
            actions=torch.zeros(batch_size, seq_len, action_dim),
            rewards=torch.zeros(batch_size, seq_len),
            dones=torch.zeros(batch_size, seq_len),
            h_seq=torch.zeros(batch_size, seq_len, deter_dim),
            z_seq=torch.zeros(batch_size, seq_len, stoch_dim),
            feats_post=torch.zeros(batch_size, seq_len, feat_dim),
        )

    def test_sc_only_audit_does_not_compute_remaining_steps(self):
        """只开 SC 时不应额外计算 remaining_steps"""
        auditor = ConsistencyAuditor(
            config=ConsistencyAuditorConfig(
                sc=SCConfig(enabled=True, horizons=(2,)),
                strict_auxiliary_losses=False,
            ),
            sc_module=nn.Identity(),
        )
        trajectory = self._make_trajectory()

        with mock.patch(
            "aletheia.aletheia_world_model.RemainingStepsComputer.compute_from_dones",
            side_effect=AssertionError("SC-only audit should not compute remaining_steps"),
        ):
            report = auditor.audit(trajectory, world_model_ref=None)

        self.assertTrue(torch.isfinite(report.sc_loss))

    def test_msc_enabled_audit_still_computes_remaining_steps(self):
        """开 MSC 时仍应保留 remaining_steps 计算路径"""
        auditor = ConsistencyAuditor(
            config=ConsistencyAuditorConfig(msc=MSCConfig(enabled=True)),
            msc_head=nn.Identity(),
        )
        trajectory = self._make_trajectory()
        remaining_steps = torch.zeros_like(trajectory.dones, dtype=torch.long)

        with mock.patch(
            "aletheia.aletheia_world_model.RemainingStepsComputer.compute_from_dones",
            return_value=remaining_steps,
        ) as remaining_mock, mock.patch.object(
            auditor, "_compute_msc", return_value=torch.tensor(0.0)
        ), mock.patch.object(
            auditor, "_compute_nst", return_value=torch.tensor(0.0)
        ), mock.patch.object(
            auditor, "_compute_mhc", return_value=torch.tensor(0.0)
        ), mock.patch.object(
            auditor, "_compute_sc", return_value=torch.tensor(0.0)
        ):
            auditor.audit(trajectory, world_model_ref=None)

        remaining_mock.assert_called_once()

    def test_nst_audit_uses_horizon_prob_dict_and_reports_active_horizons(self):
        """NST 应消费按 horizon 编号的终止概率字典，而不是静默退化为 0"""
        torch.manual_seed(0)
        msc_cfg = MSCConfig(enabled=True, horizons=(1, 4, 8), hidden_dim=16)
        nst_cfg = NSTConfig(
            enabled=True,
            n_steps=(1, 4, 8),
            loss_scale=1.0,
            use_exponential_weighting=False,
        )
        auditor = ConsistencyAuditor(
            config=ConsistencyAuditorConfig(msc=msc_cfg, nst=nst_cfg),
            msc_head=MultiScaleContinue(feat_dim=6, config=msc_cfg),
            nst_module=NStepTerminationLoss(nst_cfg),
        )
        trajectory = self._make_trajectory()
        trajectory.remaining_steps = torch.tensor(
            [[0, 2, 6, 9], [1, 4, 8, 12]],
            dtype=torch.long,
        )

        report = auditor.audit(trajectory, world_model_ref=None)

        self.assertGreater(float(report.nst_loss.item()), 0.0)
        self.assertIn("nst_active_horizons", report.metrics)
        self.assertAlmostEqual(
            float(report.metrics["nst_active_horizons"].item()),
            3.0,
            places=6,
        )


class _RaisingMHCHead(nn.Module):
    def loss(self, features, continues):
        del features, continues
        raise RuntimeError("mhc boom")


class _RaisingMSCHead(nn.Module):
    def compute_loss(self, features, remaining_steps, true_danger=None):
        del features, remaining_steps, true_danger
        raise RuntimeError("msc boom")


class _TerminationMSCHead(nn.Module):
    def forward(self, features):
        return {
            "termination_prob_dict": {
                1: torch.full(features.shape[:-1], 0.5, device=features.device),
            }
        }


class _RaisingNSTModule(nn.Module):
    def forward(self, termination_probs, remaining_steps):
        del termination_probs, remaining_steps
        raise RuntimeError("nst boom")


class _RaisingSCModule(nn.Module):
    def compute_loss(self, world_model, vitals, actions, seq_result, continues):
        del world_model, vitals, actions, seq_result, continues
        raise RuntimeError("sc boom")


class _ConstantMSCHead(nn.Module):
    def compute_loss(self, features, remaining_steps, true_danger=None):
        del remaining_steps, true_danger
        loss = features.new_tensor(1.25)
        return loss, {"total": loss.detach()}


class TestConsistencyAuditorStrictFailures(unittest.TestCase):
    def _make_trajectory(self, batch_size: int = 2, seq_len: int = 4) -> WMTrajectory:
        return TestConsistencyAuditorRemainingSteps()._make_trajectory(
            batch_size=batch_size,
            seq_len=seq_len,
        )

    def test_mhc_failure_raises_in_strict_mode_by_default(self):
        auditor = ConsistencyAuditor(
            config=ConsistencyAuditorConfig(mhc=MHCConfig(enabled=True)),
            mhc_head=_RaisingMHCHead(),
        )

        with self.assertRaisesRegex(RuntimeError, "MHC auxiliary loss failed"):
            auditor.audit(self._make_trajectory(), world_model_ref=None)

    def test_msc_failure_raises_in_strict_mode_by_default(self):
        auditor = ConsistencyAuditor(
            config=ConsistencyAuditorConfig(msc=MSCConfig(enabled=True)),
            msc_head=_RaisingMSCHead(),
        )

        with self.assertRaisesRegex(RuntimeError, "MSC auxiliary loss failed"):
            auditor.audit(self._make_trajectory(), world_model_ref=None)

    def test_nst_failure_raises_in_strict_mode_by_default(self):
        auditor = ConsistencyAuditor(
            config=ConsistencyAuditorConfig(
                msc=MSCConfig(enabled=True, horizons=(1,)),
                nst=NSTConfig(enabled=True, n_steps=(1,)),
            ),
            msc_head=_TerminationMSCHead(),
            nst_module=_RaisingNSTModule(),
        )
        trajectory = self._make_trajectory()
        trajectory.remaining_steps = torch.ones_like(trajectory.dones, dtype=torch.long)

        with mock.patch.object(auditor, "_compute_msc", return_value=torch.tensor(0.0)):
            with self.assertRaisesRegex(RuntimeError, "NST auxiliary loss failed"):
                auditor.audit(trajectory, world_model_ref=None)

    def test_sc_failure_raises_in_strict_mode_by_default(self):
        auditor = ConsistencyAuditor(
            config=ConsistencyAuditorConfig(sc=SCConfig(enabled=True, horizons=(1,))),
            sc_module=_RaisingSCModule(),
        )

        with self.assertRaisesRegex(RuntimeError, "SC auxiliary loss failed"):
            auditor.audit(self._make_trajectory(), world_model_ref=object())

    def test_mhc_failure_can_fall_back_to_zero_loss_in_non_strict_mode(self):
        auditor = ConsistencyAuditor(
            config=ConsistencyAuditorConfig(
                strict_auxiliary_losses=False,
                mhc=MHCConfig(enabled=True),
            ),
            mhc_head=_RaisingMHCHead(),
        )

        report = auditor.audit(self._make_trajectory(), world_model_ref=None)

        self.assertAlmostEqual(float(report.mhc_loss.item()), 0.0, places=6)
        self.assertTrue(bool(report.metrics["mhc_failed"]))

    def test_nst_failure_can_fall_back_to_zero_loss_in_non_strict_mode(self):
        auditor = ConsistencyAuditor(
            config=ConsistencyAuditorConfig(
                strict_auxiliary_losses=False,
                msc=MSCConfig(enabled=True, horizons=(1,)),
                nst=NSTConfig(enabled=True, n_steps=(1,)),
            ),
            msc_head=_TerminationMSCHead(),
            nst_module=_RaisingNSTModule(),
        )
        trajectory = self._make_trajectory()
        trajectory.remaining_steps = torch.ones_like(trajectory.dones, dtype=torch.long)

        with mock.patch.object(auditor, "_compute_msc", return_value=torch.tensor(0.0)):
            report = auditor.audit(trajectory, world_model_ref=None)

        self.assertAlmostEqual(float(report.nst_loss.item()), 0.0, places=6)
        self.assertTrue(bool(report.metrics["nst_failed"]))

    def test_danger_signal_failure_raises_in_strict_mode_by_default(self):
        auditor = ConsistencyAuditor(
            config=ConsistencyAuditorConfig(msc=MSCConfig(enabled=True)),
            msc_head=_ConstantMSCHead(),
        )
        auditor.set_danger_fn(lambda vitals: (_ for _ in ()).throw(RuntimeError("danger boom")))

        with self.assertRaisesRegex(RuntimeError, "Danger signal computation failed"):
            auditor.audit(self._make_trajectory(), world_model_ref=None)

    def test_danger_signal_failure_can_fall_back_in_non_strict_mode(self):
        auditor = ConsistencyAuditor(
            config=ConsistencyAuditorConfig(
                strict_auxiliary_losses=False,
                msc=MSCConfig(enabled=True),
            ),
            msc_head=_ConstantMSCHead(),
        )
        auditor.set_danger_fn(lambda vitals: (_ for _ in ()).throw(RuntimeError("danger boom")))

        report = auditor.audit(self._make_trajectory(), world_model_ref=None)

        self.assertGreater(float(report.msc_loss.item()), 0.0)

    def test_config_rejects_enabling_mhc_and_msc_together(self):
        with self.assertRaisesRegex(ValueError, "MHC and MSC cannot both be enabled"):
            ConsistencyAuditorConfig(
                mhc=MHCConfig(enabled=True),
                msc=MSCConfig(enabled=True),
            )

    def test_config_rejects_nst_without_msc_support(self):
        with self.assertRaisesRegex(ValueError, "NST requires MSC to be enabled"):
            ConsistencyAuditorConfig(
                nst=NSTConfig(enabled=True),
            )


class _FakeShortcutTransitionOut:
    def __init__(self, h: torch.Tensor, z: torch.Tensor):
        self.h = h
        self.z_prior_raw = z


class _FakeShortcutStateTransition:
    def imagine_step(self, prev_state, action_embed):
        h, z = prev_state
        return _FakeShortcutTransitionOut(h + action_embed, z + action_embed)

    def get_features(self, h: torch.Tensor, z_raw: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z_raw, h + 2.0 * z_raw], dim=-1)


class _FakeShortcutWorldModel:
    def __init__(self):
        self.state_transition = _FakeShortcutStateTransition()

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        return action.float()

    def get_features(self, h: torch.Tensor, z_raw: torch.Tensor) -> torch.Tensor:
        return self.state_transition.get_features(h, z_raw)


class TestShortcutConsistencyFeatureAlignment(unittest.TestCase):
    """测试 SC 能直接对齐 actor/critic 消费的 feature 语义"""

    def test_feature_alignment_term_increases_sc_penalty_when_enabled(self):
        world_model = _FakeShortcutWorldModel()
        vitals = torch.zeros(1, 3, 1)
        actions = torch.ones(1, 2, 1)
        continues = torch.ones(1, 2)
        h_seq = torch.tensor([[[0.0], [0.5]]])
        z_seq = torch.tensor([[[0.0], [0.5]]])
        feats_post = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.5, 1.5]]])
        seq_result = {
            "h_seq": h_seq,
            "z_seq": z_seq,
            "feats_post": feats_post,
        }

        base_loss, _ = ShortcutConsistencyLoss(
            SCConfig(enabled=True, horizons=(1,), loss_scale=1.0, feature_loss_scale=0.0)
        ).compute_loss(
            world_model=world_model,
            vitals=vitals,
            actions=actions,
            seq_result=seq_result,
            continues=continues,
        )

        feature_loss, metrics = ShortcutConsistencyLoss(
            SCConfig(enabled=True, horizons=(1,), loss_scale=1.0, feature_loss_scale=1.0)
        ).compute_loss(
            world_model=world_model,
            vitals=vitals,
            actions=actions,
            seq_result=seq_result,
            continues=continues,
        )

        self.assertGreater(float(feature_loss.item()), float(base_loss.item()))
        self.assertIn("feature_loss_h1", metrics)


# =========================================================================
# TEST-5: WorldModel bridge contract
# =========================================================================


class TestWorldModelBridgeContract(unittest.TestCase):
    def test_compute_loss_keeps_bridge_losses_alive_behind_full_wall(self):
        device = torch.device("cpu")

        rssm_cfg = RSSMConfig(
            vitals_dim=4,
            deter_dim=8,
            hidden_dim=16,
            action_embed_dim=8,
            obs_embed_dim=8,
            z_embed_dim=8,
            distribution=DistributionConfig(
                num_distributions=2,
                num_classes=4,
                use_unimix=False,
            ),
            kl=KLConfig(free_nats=0.0),
            continue_config=ContinueConfig(hidden_dims=[16]),
            use_feature_norm=True,
            use_separate_norm=False,
            use_persist_gate=False,
        )
        wm_cfg = WorldModelConfig(
            obs_embed_dim=8,
            deter_dim=8,
            stoch_dim=8,
            action_dim=2,
            num_classes=4,
            control=ControlConfig(
                mode="static",
                ctrl_dim=6,
                hidden_dims=(16,),
                align_strength=1.0,
                reg_strength=1.0,
            ),
            projection=ProjectionConfig(
                enabled=True,
                hidden_dim=16,
                mmd_weight=0.0,
                mmd_mode="none",
            ),
            consistency_head=ConsistencyHeadConfig(
                enabled=True,
                hidden_dim=16,
            ),
            aux_tasks=AuxTasksConfig(
                enabled=True,
                hidden_dim=16,
                value_dim=1,
            ),
            wall=GradientWallConfig(
                warmup_steps=1,
                initial_wall_strength=1.0,
                final_wall_strength=1.0,
                bridge_maintenance_floor=0.4,
            ),
            loss_weights=LossWeightsConfig(
                recon=0.0,
                reward=0.0,
                continue_=0.0,
                kl=0.0,
                quant=0.0,
                projection=1.0,
                control_align=1.0,
                control_reg=0.0,
                abstractor=0.0,
                consistency_head=1.0,
                aux_inverse=1.0,
                aux_value=1.0,
            ),
        )

        world_model = WorldModel(rssm_cfg, wm_config=wm_cfg).to(device)
        world_model.setup_action_space(
            SimpleNamespace(
                shape=(2,),
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
            )
        )

        torch.manual_seed(3)
        batch_size, seq_len = 2, 3
        vitals = torch.randn(batch_size, seq_len + 1, 4, device=device)
        actions = torch.tanh(torch.randn(batch_size, seq_len, 2, device=device))
        rewards = torch.randn(batch_size, seq_len, device=device)
        dones = torch.zeros(batch_size, seq_len, device=device)

        seq_result = world_model.observe_sequence(
            vitals=vitals,
            actions=actions,
            dones=dones,
        )
        trajectory = world_model.build_trajectory(
            seq_result=seq_result,
            vitals=vitals,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )
        loss_packet = world_model.compute_loss(trajectory)

        self.assertAlmostEqual(
            float(loss_packet.metrics["bridge/maintenance_scale"]),
            0.4,
            places=6,
        )
        self.assertGreater(float(loss_packet.projection.detach().item()), 0.0)
        self.assertGreater(float(loss_packet.control_align.detach().item()), 0.0)
        self.assertGreater(float(loss_packet.consistency_head.detach().item()), 0.0)
        self.assertGreater(float(loss_packet.aux_inverse.detach().item()), 0.0)
        self.assertGreater(float(loss_packet.aux_value.detach().item()), 0.0)
        self.assertGreater(float(loss_packet.total.detach().item()), 0.0)


class TestWorldModelRssmAlignmentContracts(unittest.TestCase):
    def _make_alignment_subject(
        self,
        *,
        rssm_deter_dim: int = 8,
        rssm_obs_embed_dim: int = 8,
        rssm_num_distributions: int = 2,
        rssm_num_classes: int = 4,
        wm_deter_dim: int = 8,
        wm_obs_embed_dim: int = 8,
        wm_stoch_dim: int = 8,
        wm_num_classes: int = 4,
    ):
        rssm_cfg = RSSMConfig(
            vitals_dim=4,
            deter_dim=rssm_deter_dim,
            hidden_dim=16,
            action_embed_dim=8,
            obs_embed_dim=rssm_obs_embed_dim,
            z_embed_dim=8,
            distribution=DistributionConfig(
                num_distributions=rssm_num_distributions,
                num_classes=rssm_num_classes,
                use_unimix=False,
            ),
            kl=KLConfig(free_nats=0.0),
            continue_config=ContinueConfig(hidden_dims=[16]),
            use_feature_norm=True,
            use_separate_norm=False,
            use_persist_gate=False,
        )
        wm_cfg = WorldModelConfig(
            obs_embed_dim=wm_obs_embed_dim,
            deter_dim=wm_deter_dim,
            stoch_dim=wm_stoch_dim,
            action_dim=2,
            num_classes=wm_num_classes,
        )
        world_model = WorldModel.__new__(WorldModel)
        world_model.config = rssm_cfg
        world_model._wm_config = wm_cfg
        return world_model, rssm_cfg, wm_cfg

    def test_align_rssm_config_rejects_explicit_field_mismatch_across_validation_modes(self):
        for validation_mode in ("strict", "warn", "off"):
            with self.subTest(validation_mode=validation_mode):
                world_model, _, _ = self._make_alignment_subject(
                    rssm_deter_dim=10,
                    wm_deter_dim=8,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "auto-alignment is no longer supported",
                ):
                    WorldModel._align_rssm_config(
                        world_model,
                        validation_mode=validation_mode,
                    )

    def test_align_rssm_config_rejects_num_distribution_mismatch_across_validation_modes(self):
        for validation_mode in ("strict", "warn", "off"):
            with self.subTest(validation_mode=validation_mode):
                world_model, _, _ = self._make_alignment_subject(
                    rssm_num_distributions=3,
                    wm_stoch_dim=8,
                    wm_num_classes=4,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "RSSMConfig.num_distributions",
                ):
                    WorldModel._align_rssm_config(
                        world_model,
                        validation_mode=validation_mode,
                    )

    def test_align_rssm_config_rejects_nondivisible_world_model_stoch_dim(self):
        for validation_mode in ("strict", "warn", "off"):
            with self.subTest(validation_mode=validation_mode):
                world_model, _, _ = self._make_alignment_subject()
                world_model._wm_config = SimpleNamespace(
                    obs_embed_dim=8,
                    deter_dim=8,
                    stoch_dim=10,
                    num_classes=4,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "not divisible",
                ):
                    WorldModel._align_rssm_config(
                        world_model,
                        validation_mode=validation_mode,
                    )

    def test_align_rssm_config_allows_consistent_config(self):
        world_model, rssm_cfg, _ = self._make_alignment_subject()

        WorldModel._align_rssm_config(world_model, validation_mode="off")

        self.assertEqual(int(rssm_cfg.deter_dim), 8)
        self.assertEqual(int(rssm_cfg.obs_embed_dim), 8)
        self.assertEqual(int(rssm_cfg.distribution.num_classes), 4)
        self.assertEqual(int(rssm_cfg.distribution.num_distributions), 2)


class TestWorldModelResidualWarningContracts(unittest.TestCase):
    def test_mhc_short_sequence_remains_diagnostic_warning(self):
        mhc = MultiHorizonContinueHead(
            MultiHorizonContinueConfig(
                feat_dim=4,
                horizons=(1, 3),
                hidden_dim=8,
                num_layers=1,
                min_seq_len=5,
                skip_invalid_horizons=True,
                loss_scale=1.0,
            )
        )
        feats = torch.randn(2, 2, 4)
        continues = torch.ones(2, 2)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loss, metrics = mhc.loss(feats, continues)

        self.assertEqual(len(caught), 1)
        self.assertIn("Sequence length 2 < min_seq_len 5", str(caught[0].message))
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(metrics["mhc_valid_horizons"]), 1.0, places=6)

    def test_prediction_heads_reject_mhc_feat_dim_mismatch(self):
        config = PredictionHeadsConfig(
            feat_dim=8,
            vitals_dim=4,
            mhc=MultiHorizonContinueConfig(
                enabled=True,
                feat_dim=6,
                horizons=(1, 3),
                hidden_dim=8,
                num_layers=1,
                loss_scale=1.0,
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "MHC feat_dim",
        ):
            PredictionHeads(config)

    def test_apply_wall_strength_rejects_router_weight_shape_drift(self):
        features = PolicyFeatures(
            f_policy=torch.randn(1, 4),
            v_x=torch.randn(1, 4),
            v_c=torch.randn(1, 4),
            v_z=torch.randn(1, 4),
            weights=torch.tensor([[0.7, 0.3]], dtype=torch.float32),
            uncertainty=torch.zeros(1),
            mask=torch.zeros(1),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Expected router to produce 3 branch weights",
        ):
            _apply_wall_strength_to_policy_features(features, wall_strength=0.5)


# =========================================================================
# TEST-6: compute_lambda_returns & symlog/symexp 一致性
# =========================================================================

class TestComputeLambdaReturns(unittest.TestCase):
    """测试 -returns 计算"""

    def test_single_gamma_output_shape(self):
        """单 gamma  输出与 values 同形"""
        T, B = 10, 4
        rewards = torch.randn(T, B)
        values = torch.randn(T, B)
        bootstrap = torch.randn(B)

        ret = compute_lambda_returns(rewards, values, bootstrap, gamma=0.99, lambda_=0.95)
        self.assertEqual(ret.shape, (T, B))
        self.assertTrue(torch.isfinite(ret).all())

    def test_multi_gamma_output_shape(self):
        """多 gamma  输出末维堆叠"""
        T, B = 10, 4
        gammas = [0.9, 0.95, 0.99]
        rewards = torch.randn(T, B)
        values = torch.randn(T, B)
        bootstrap = torch.randn(B)

        ret = compute_lambda_returns(rewards, values, bootstrap, gamma=gammas, lambda_=0.95)
        self.assertEqual(ret.shape, (T, B, len(gammas)))

    def test_zero_reward_matches_values(self):
        """reward=0, gamma=0 时returns 应约等于 values"""
        T, B = 5, 2
        values = torch.randn(T, B)
        rewards = torch.zeros(T, B)
        bootstrap = torch.zeros(B)

        ret = compute_lambda_returns(rewards, values, bootstrap, gamma=0.0, lambda_=0.95)
        # gamma=0 => delta[t] = 0 + 0*next_val - values[t] = -values[t]
        # returns[t] = gae + values[t] = -values[t] + values[t] = 0
        self.assertTrue(torch.allclose(ret, torch.zeros_like(ret), atol=1e-5))

    def test_lambda_zero_equals_one_step_return(self):
        """lambda=0 时returns 应等于 r + *V(s_{t+1})"""
        T, B = 5, 2
        gamma = 0.99
        rewards = torch.ones(T, B)
        values = torch.ones(T, B) * 2
        bootstrap = torch.ones(B) * 2

        ret = compute_lambda_returns(rewards, values, bootstrap, gamma=gamma, lambda_=0.0)
        # lambda=0 => GAE = delta[t] (one-step)
        # delta[t] = r + gamma * V(s_{t+1}) - V(s_t)
        # returns[t] = delta[t] + V(s_t) = r + gamma * V(s_{t+1})
        expected = rewards + gamma * torch.cat([values[1:], bootstrap.unsqueeze(0)], dim=0)
        self.assertTrue(torch.allclose(ret, expected, atol=1e-5))


class TestSymlogSymexpConsistency(unittest.TestCase):
    """测试 symlog/symexp 的一致性和统一性"""

    def test_symlog_symexp_roundtrip(self):
        """symlog  symexp 应近似还原"""
        x = torch.randn(100) * 10
        reconstructed = symexp(symlog(x))
        self.assertTrue(torch.allclose(reconstructed, x, atol=1e-4))

    def test_symlog_zero(self):
        """symlog(0)  0"""
        z = torch.zeros(1)
        self.assertTrue(torch.allclose(symlog(z), torch.zeros(1), atol=1e-6))

    def test_symlog_symmetry(self):
        """symlog(-x) = -symlog(x)"""
        x = torch.randn(50).abs() + 1.0
        self.assertTrue(torch.allclose(symlog(-x), -symlog(x), atol=1e-6))

    def test_actor_critic_symlog_is_foundation_symlog(self):
        """actor_critic 的 symlog/symexp 应与 foundation 版本完全一致"""
        x = torch.randn(100) * 5
        self.assertTrue(torch.equal(ac_symlog(x), symlog(x)))
        self.assertTrue(torch.equal(ac_symexp(x), symexp(x)))


class TestRSSMConfigContracts(unittest.TestCase):
    def test_categorical_rssm_rejects_distribution_dim_mismatch(self):
        with self.assertRaisesRegex(
            ValueError,
            "RSSMConfig: distribution dims mismatch",
        ):
            RSSMConfig(
                vitals_dim=4,
                deter_dim=8,
                hidden_dim=16,
                action_embed_dim=8,
                obs_embed_dim=8,
                z_embed_dim=8,
                z_type="categorical",
                z_categories=3,
                z_classes=5,
                distribution=DistributionConfig(
                    num_distributions=2,
                    num_classes=4,
                    use_unimix=False,
                ),
            )

    def test_categorical_rssm_still_backfills_missing_distribution_dims(self):
        cfg = RSSMConfig(
            vitals_dim=4,
            deter_dim=8,
            hidden_dim=16,
            action_embed_dim=8,
            obs_embed_dim=8,
            z_embed_dim=8,
            z_type="categorical",
            z_categories=0,
            z_classes=0,
            distribution=DistributionConfig(
                num_distributions=2,
                num_classes=4,
                use_unimix=False,
            ),
        )

        self.assertEqual(int(cfg.z_categories), 2)
        self.assertEqual(int(cfg.z_classes), 4)
        self.assertEqual(int(cfg.distribution.num_distributions), 2)
        self.assertEqual(int(cfg.distribution.num_classes), 4)


if __name__ == "__main__":
    unittest.main()
