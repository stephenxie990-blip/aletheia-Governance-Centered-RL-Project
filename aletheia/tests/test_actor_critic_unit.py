"""Unit tests for aletheia_actor_critic module components.

Migrated from inline _test_* functions in aletheia_actor_critic.py.
"""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from aletheia.aletheia_actor_critic import (
    Actor,
    ActorConfig,
    ActorHealthStats,
    ActorCritic,
    AdaptiveGammaRouter,
    CategoricalStraightThrough,
    CriticConfig,
    CriticOutput,
    CuriosityModule,
    DoubleCritic,
    EntropyProtectionConfig,
    HealthMonitor,
    HierarchicalUnifiedCritic,
    MasteryModule,
    SingleCritic,
    TanhTransformedDistribution,
    TrustModel,
    ValueNormalizer,
    Will,
    WillConfig,
    WillInputs,
    create_actor_critic,
    create_will,
    create_will_optimizer,
)


class TestActorBasic(unittest.TestCase):
    def test_discrete_actor(self):
        B = 4
        actor = Actor(feat_dim=64, action_dim=5, is_discrete=True)
        dist = actor(torch.randn(B, 64))
        action = dist.rsample()
        self.assertEqual(action.shape, (B, 5))
        lp = dist.log_prob(action)
        self.assertEqual(lp.shape, (B,))
        ent = dist.entropy()
        self.assertEqual(ent.shape, (B,))

    def test_continuous_actor(self):
        B = 4
        actor = Actor(feat_dim=64, action_dim=3, is_discrete=False)
        dist = actor(torch.randn(B, 64))
        action = dist.rsample()
        self.assertEqual(action.shape, (B, 3))
        self.assertTrue((action.abs() <= 1.0).all())
        lp = dist.log_prob(action)
        self.assertEqual(lp.shape, (B,))


class TestActorEntropyProtection(unittest.TestCase):
    def test_entropy_protection(self):
        actor = Actor(
            feat_dim=64,
            action_dim=4,
            is_discrete=True,
            config=ActorConfig(
                entropy_protection=EntropyProtectionConfig(enabled=True),
            ),
        )
        self.assertIsNotNone(actor.entropy_protector)
        dist = actor(torch.randn(2, 64))
        ent = dist.entropy()
        loss, info = actor.entropy_loss(ent)
        self.assertIn("entropy", info)
        self.assertIn("alpha", info)
        alpha_loss = actor.alpha_loss(ent)
        self.assertEqual(alpha_loss.shape, (1,))
        actor.step_entropy_target()
        health = actor.health_stats()
        self.assertIsInstance(health, ActorHealthStats)


class TestHealthMonitor(unittest.TestCase):
    def test_ema_behaviour(self):
        mon = HealthMonitor(ema_alpha=0.1)
        for _ in range(50):
            mon.record_logits_clip(80, 100)
        high = mon.logits_clip_rate
        for _ in range(50):
            mon.record_logits_clip(0, 100)
        low = mon.logits_clip_rate
        self.assertLess(low, high)
        mon.record_loc_clip(10, 100)
        mon.record_scale_clip(5, 100)
        self.assertGreater(mon.loc_clip_rate, 0)
        self.assertGreater(mon.scale_clip_rate, 0)
        self.assertNotEqual(mon.loc_clip_rate, mon.scale_clip_rate)


class TestCriticSingle(unittest.TestCase):
    def test_single_critic(self):
        c = SingleCritic(feat_dim=64, primary_gamma=0.99)
        feat = torch.randn(4, 64)
        out = c(feat)
        self.assertEqual(out.value.shape, (4,))
        self.assertEqual(out.uncertainty.shape, (4,))
        out2 = c(feat, use_target=True)
        self.assertEqual(out2.value.shape, (4,))
        loss_d = c.compute_loss(feat, torch.randn(4))
        self.assertIn("loss", loss_d)
        loss_d2 = c.compute_loss(feat, torch.randn(4), detach_features=True)
        self.assertIn("loss", loss_d2)


class TestCriticOutputGetitem(unittest.TestCase):
    def test_getitem(self):
        out = CriticOutput(
            values_symlog_main=torch.zeros(4),
            values_real_main=torch.zeros(4),
            uncertainty_raw=torch.zeros(4),
        )
        _ = out["value"]
        _ = out["uncertainty"]
        with self.assertRaises(KeyError):
            _ = out["nonexistent_key"]


class TestDoubleCritic(unittest.TestCase):
    def test_no_double_pessimism(self):
        cfg = CriticConfig(
            d_feature=64,
            gammas=(0.99,),
            mode="double",
            use_target_critic=True,
            pessimism=0.25,
        )
        dc = DoubleCritic(cfg)
        self.assertEqual(dc.c1.pessimism, 0.5)
        self.assertEqual(dc.c2.pessimism, 0.5)
        feat = torch.randn(4, 64)
        out1, out2 = dc.forward(feat, use_target=False)
        self.assertEqual(out1.value.shape, (4,))
        out_min = dc.forward_min(feat, use_target=True)
        self.assertEqual(out_min.value.shape, (4,))
        targets = {0.99: torch.randn(4)}
        ld = dc.compute_loss(feat, targets, detach_features=True)
        self.assertIn("loss", ld)


class TestCuriosityDimensionFix(unittest.TestCase):
    def test_all_dimension_combos(self):
        config = WillConfig()
        cur = CuriosityModule(config)
        r1 = cur(torch.rand(4), torch.rand(4), F.softmax(torch.randn(4, 3), dim=-1))
        self.assertEqual(r1.shape, (4,))
        r2 = cur(torch.rand(4, 10), torch.rand(4, 10), F.softmax(torch.randn(4, 3), dim=-1))
        self.assertEqual(r2.shape, (4, 10))
        r3 = cur(torch.rand(4, 10), torch.rand(4, 10), F.softmax(torch.randn(4, 10, 3), dim=-1))
        self.assertEqual(r3.shape, (4, 10))


class TestTrustBufferPreservation(unittest.TestCase):
    def test_buffers_preserved(self):
        config = WillConfig()
        trust = TrustModel(config)
        self.assertIn("trust_wm", dict(trust.named_buffers()))
        self.assertIn("trust_v", dict(trust.named_buffers()))
        self.assertIn("trust_combined", dict(trust.named_buffers()))
        trust.update(torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.8]))
        self.assertIn("trust_wm", dict(trust.named_buffers()))
        self.assertIn("trust_v", dict(trust.named_buffers()))
        self.assertIn("trust_combined", dict(trust.named_buffers()))
        sd = trust.state_dict()
        self.assertIn("trust_wm", sd)
        self.assertIn("trust_v", sd)
        self.assertIn("trust_combined", sd)


class TestMasteryUnified(unittest.TestCase):
    def test_mastery(self):
        config = WillConfig()
        mastery = MasteryModule(config)
        B, T = 4, 10
        U_now = torch.rand(B)
        U_prev = U_now + 0.5
        r = mastery(U_now, U_now, U_prev, U_prev)
        self.assertTrue((r >= 0).all())
        self.assertEqual(r.shape, (B,))
        U_dec = torch.linspace(2.0, 0.5, T).unsqueeze(0).expand(B, T)
        r_seq = mastery(U_dec, U_dec)
        self.assertEqual(r_seq.shape, (B, T))
        self.assertTrue((r_seq[:, 0] == 0).all())


class TestWillBasic(unittest.TestCase):
    def test_will_basic(self):
        config = WillConfig()
        will = Will(config)
        B, T = 4, 10
        inputs = WillInputs(
            U_wm=torch.rand(B, T),
            U_v=torch.rand(B, T) * 0.5,
            router_weights=F.softmax(torch.randn(B, 3), dim=-1),
            policy_entropy=torch.rand(B, T) * 2.0,
        )
        out = will(inputs, step=0)
        self.assertEqual(out.r_intrinsic.shape, (B, T))
        self.assertEqual(out.r_curiosity.shape, (B, T))
        self.assertGreaterEqual(out.trust, 0.0)
        self.assertLessEqual(out.trust, 1.0)


class TestWillTrustMonotonicity(unittest.TestCase):
    def test_trust_monotonicity(self):
        config = WillConfig(trust_ema_beta=0.5)
        will = Will(config)
        B = 4
        high_inp = WillInputs(
            U_wm=torch.ones(B) * 5.0,
            U_v=torch.ones(B) * 3.0,
            router_weights=F.softmax(torch.randn(B, 3), dim=-1),
            policy_entropy=torch.rand(B),
        )
        for i in range(20):
            out_h = will(high_inp, step=i)
        trust_high = out_h.trust

        will.reset()

        low_inp = WillInputs(
            U_wm=torch.ones(B) * 0.1,
            U_v=torch.ones(B) * 0.05,
            router_weights=F.softmax(torch.randn(B, 3), dim=-1),
            policy_entropy=torch.rand(B),
        )
        for i in range(20):
            out_l = will(low_inp, step=i)
        trust_low = out_l.trust

        self.assertGreater(trust_low, trust_high)


class TestWillComponents(unittest.TestCase):
    def test_components(self):
        config = WillConfig(
            curiosity_enabled=True,
            mastery_enabled=True,
            autonomy_enabled=True,
            trust_enabled=True,
        )
        will = Will(config)
        B, T = 2, 8
        inputs = WillInputs(
            U_wm=torch.rand(B, T),
            U_v=torch.rand(B, T),
            router_weights=F.softmax(torch.randn(B, T, 3), dim=-1),
            policy_entropy=torch.rand(B, T),
            U_wm_prev=torch.rand(B, T) + 0.1,
            U_v_prev=torch.rand(B, T) + 0.1,
        )
        out = will(inputs)
        self.assertEqual(out.r_curiosity.shape, (B, T))
        self.assertEqual(out.r_intrinsic.shape, (B, T))
        self.assertIsNotNone(out.trust_wm)
        self.assertIsNotNone(out.trust_v)


class TestWillGradientIsolation(unittest.TestCase):
    def test_gradient_isolation(self):
        config = WillConfig()
        will = Will(config)
        B = 4
        U_wm = torch.rand(B, requires_grad=True)
        U_v = torch.rand(B, requires_grad=True)
        router_w = F.softmax(torch.randn(B, 3, requires_grad=True), dim=-1)
        policy_ent = torch.rand(B, requires_grad=True)
        out = will(
            WillInputs(U_wm=U_wm, U_v=U_v, router_weights=router_w, policy_entropy=policy_ent),
            step=0,
        )
        self.assertIsNone(out.r_intrinsic.grad_fn)
        self.assertFalse(out.r_intrinsic.requires_grad)
        self.assertTrue(U_wm.requires_grad)


class TestActorCriticForward(unittest.TestCase):
    def test_forward_signature(self):
        class FakeDiscreteSpace:
            n = 4

        ac = create_actor_critic(feat_dim=64, action_space=FakeDiscreteSpace())
        feat = torch.randn(2, 64)
        dist, value = ac(feat, temperature=0.5)
        action = dist.rsample()
        lp = dist.log_prob(action)
        self.assertEqual(lp.shape, (2,))
        self.assertEqual(value.shape, (2,))
        critic_out = ac.get_value(feat)
        self.assertIsInstance(critic_out, CriticOutput)
        critic_out_tgt = ac.get_value(feat, use_target=True)
        self.assertIsInstance(critic_out_tgt, CriticOutput)
        ac.update_target()


class TestValueNormalizer(unittest.TestCase):
    def test_normalizer(self):
        vn = ValueNormalizer(beta=0.99)
        for _ in range(50):
            returns = torch.randn(32) * 100.0 + 50.0
            vn.normalize(returns)
        test_returns = torch.randn(100) * 100.0 + 50.0
        normed = vn.normalize(test_returns, update=False)
        self.assertLess(abs(normed.mean().item()), 5.0)
        recovered = vn.denormalize(normed)
        err = (recovered - test_returns).abs().mean().item()
        self.assertLess(err, 1e-4)
        sd = vn.state_dict()
        self.assertIn("_mean", sd)
        self.assertIn("_var", sd)


class TestRouterDetach(unittest.TestCase):
    def test_gradient_isolation(self):
        cfg = CriticConfig(d_feature=64, gammas=(0.9, 0.99, 0.999), adaptive_hidden_dim=32)
        router = AdaptiveGammaRouter(cfg)
        feat = torch.randn(4, 64, requires_grad=True)
        values = {g: torch.randn(4, requires_grad=True) for g in cfg.gammas}
        targets = {g: torch.randn(4) for g in cfg.gammas}
        rl = router.compute_router_loss(feat, values, targets, cfg.gammas)
        self.assertTrue(rl["L_route"].requires_grad or rl["L_route"].grad_fn is not None)
        rl["L_route"].backward(retain_graph=True)
        self.assertIsNone(feat.grad)


class TestHierarchicalCriticLossSeparation(unittest.TestCase):
    def test_loss_separation(self):
        cfg = CriticConfig(
            d_feature=64,
            gammas=(0.9, 0.99, 0.999),
            mode="ensemble",
            use_adaptive_routing=True,
            adaptive_hidden_dim=32,
        )
        hc = HierarchicalUnifiedCritic(cfg)
        feat = torch.randn(4, 64)
        targets = {g: torch.randn(4) for g in cfg.gammas}
        result = hc.compute_loss(feat, targets)
        self.assertIn("loss", result)
        self.assertIn("loss_critic", result)
        if hc.router is not None:
            self.assertIn("loss_router", result)


class TestEntropyCache(unittest.TestCase):
    def test_categorical_cache(self):
        logits = torch.randn(4, 5)
        cat = CategoricalStraightThrough(logits)
        e1 = cat.entropy()
        e2 = cat.entropy()
        self.assertIs(e1, e2)

    def test_tanh_cache(self):
        loc = torch.randn(4, 3)
        scale = torch.ones(4, 3) * 0.5
        tanh = TanhTransformedDistribution(loc, scale)
        e1 = tanh.entropy()
        e2 = tanh.entropy()
        self.assertIs(e1, e2)


class TestFactory(unittest.TestCase):
    def test_create_will(self):
        will = create_will()
        self.assertIsInstance(will, Will)
        cfg = WillConfig()
        will2 = create_will(cfg)
        self.assertIsInstance(will2, Will)
        opt = create_will_optimizer(will)


if __name__ == "__main__":
    unittest.main()
