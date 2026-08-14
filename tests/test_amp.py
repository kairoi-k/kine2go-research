"""CPU-safe regression tests for conditional-AMP utilities."""

import unittest

import numpy as np
import torch

from motion_imitation.amp import (
    AMPDiscriminator,
    BoxAdaptiveCurriculum,
    bucket_balanced_sample,
    r1_gradient_penalty,
    style_reward_from_logit,
)


class FakeEnv:
    def __init__(self):
        self.command_cfg = {"lin_vel_x_range": (0.9, 1.1)}


class AMPUtilityTests(unittest.TestCase):
    def test_style_reward_scales_with_timestep(self):
        logits = torch.zeros(4)
        reward = style_reward_from_logit(logits, scale=0.3, dt=1.0 / 60.0)
        expected = torch.full((4,), 0.5 * 0.3 / 60.0)
        self.assertTrue(torch.allclose(reward, expected))

    def test_discriminator_output_shape(self):
        disc = AMPDiscriminator(state_dim=30, hidden=32)
        s = torch.randn(8, 30)
        sp = torch.randn(8, 30)
        c = torch.rand(8)
        self.assertEqual(tuple(disc(s, sp, c).shape), (8,))

    def test_bucket_sampler_returns_requested_batch(self):
        ref_c = torch.cat([
            torch.linspace(0.3, 0.79, 600),
            torch.linspace(0.8, 1.19, 120),
            torch.linspace(1.2, 1.59, 120),
            torch.linspace(1.6, 2.09, 120),
            torch.linspace(2.1, 2.69, 120),
            torch.linspace(2.7, 3.49, 120),
        ])
        ref_s = torch.randn(len(ref_c), 30)
        ref_sp = torch.randn(len(ref_c), 30)
        rng = np.random.default_rng(7)
        s, sp, c = bucket_balanced_sample(
            ref_s,
            ref_sp,
            ref_c,
            batch=240,
            rng=rng,
        )
        self.assertEqual(len(s), 240)
        self.assertEqual(len(sp), 240)
        self.assertEqual(len(c), 240)
        self.assertGreaterEqual(float(c.min()), 0.3)
        self.assertLess(float(c.max()), 3.5)

    def test_r1_penalty_is_finite_and_nonnegative(self):
        torch.manual_seed(0)
        disc = AMPDiscriminator(state_dim=30, hidden=32)
        s = torch.randn(16, 30)
        sp = torch.randn(16, 30)
        c = torch.rand(16)
        penalty = r1_gradient_penalty(disc, s, sp, c, lambda_gp=10.0)
        self.assertTrue(torch.isfinite(penalty))
        self.assertGreaterEqual(float(penalty), 0.0)

    def test_curriculum_respects_target_and_support(self):
        env = FakeEnv()
        curriculum = BoxAdaptiveCurriculum(
            env,
            start_range=(0.9, 1.1),
            target_range=(0.5, 2.5),
            step=0.1,
            eval_interval=10,
            success_threshold=0.6,
            support_range=(0.3, 3.5),
        )

        for i in range(40):
            curriculum.record(
                torch.tensor([1.0]),
                torch.tensor([1.0]),
                torch.tensor([True]),
            )
            curriculum.step_update((i + 1) * 10)

        lo, hi = env.command_cfg["lin_vel_x_range"]
        self.assertGreaterEqual(lo, 0.5)
        self.assertLessEqual(hi, 2.5)
        self.assertGreaterEqual(lo, 0.3)
        self.assertLessEqual(hi, 3.5)


if __name__ == "__main__":
    unittest.main()
