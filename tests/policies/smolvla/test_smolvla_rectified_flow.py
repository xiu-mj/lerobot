import pytest
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    consistency_ema_decay,
    consistency_velocity_target,
    flow_solver_step,
    flow_training_path,
    sample_flow_time,
)


def test_smolvla_config_defaults_to_rectified_flow():
    cfg = SmolVLAConfig()

    assert cfg.flow_objective == "rectified_flow"
    assert cfg.flow_time_sampling == "uniform"
    assert cfg.flow_solver == "euler"


def test_rectified_flow_training_path_uses_noise_to_action_direction():
    actions = torch.tensor([[[2.0, 4.0]]])
    noise = torch.tensor([[[0.0, 10.0]]])
    time = torch.tensor([0.25])

    x_t, u_t = flow_training_path(actions, noise, time, objective="rectified_flow")

    assert torch.allclose(x_t, torch.tensor([[[0.5, 8.5]]]))
    assert torch.allclose(u_t, torch.tensor([[[2.0, -6.0]]]))


def test_legacy_flow_matching_training_path_is_available():
    actions = torch.tensor([[[2.0, 4.0]]])
    noise = torch.tensor([[[0.0, 10.0]]])
    time = torch.tensor([0.25])

    x_t, u_t = flow_training_path(actions, noise, time, objective="flow_matching")

    assert torch.allclose(x_t, torch.tensor([[[1.5, 5.5]]]))
    assert torch.allclose(u_t, torch.tensor([[[-2.0, 6.0]]]))


def test_flow_time_sampling_respects_epsilon_bounds():
    time = sample_flow_time(
        bsize=256,
        device=torch.device("cpu"),
        sampling="uniform",
        beta_alpha=1.5,
        beta_beta=1.0,
        eps=0.01,
    )

    assert time.shape == (256,)
    assert torch.all(time >= 0.01)
    assert torch.all(time <= 0.99)


def test_invalid_flow_objective_raises():
    actions = torch.zeros(1, 1, 1)
    noise = torch.ones(1, 1, 1)
    time = torch.zeros(1)

    with pytest.raises(ValueError):
        flow_training_path(actions, noise, time, objective="bad-objective")


def test_euler_solver_step_uses_current_velocity():
    x_t = torch.tensor([1.0])
    v_t = torch.tensor([2.0])

    x_next, step_velocity = flow_solver_step(x_t, v_t, dt=0.25, solver="euler")

    assert torch.allclose(x_next, torch.tensor([1.5]))
    assert torch.equal(step_velocity, v_t)


def test_heun_solver_step_averages_predictor_and_corrector_velocities():
    x_t = torch.tensor([1.0])
    v_t = torch.tensor([2.0])
    v_next = torch.tensor([4.0])

    x_next, step_velocity = flow_solver_step(x_t, v_t, dt=0.25, solver="heun", v_next=v_next)

    assert torch.allclose(step_velocity, torch.tensor([3.0]))
    assert torch.allclose(x_next, torch.tensor([1.75]))


def test_heun_solver_requires_corrector_velocity():
    with pytest.raises(ValueError, match="Heun solver requires"):
        flow_solver_step(torch.zeros(1), torch.ones(1), dt=0.1, solver="heun")


def test_invalid_flow_solver_config_raises():
    with pytest.raises(ValueError, match="flow_solver"):
        SmolVLAConfig(flow_solver="bad-solver")


def test_consistency_velocity_target_uses_teacher_endpoint():
    x_t = torch.tensor([[[2.0]]])
    x_t_next = torch.tensor([[[3.0]]])
    velocity_next = torch.tensor([[[4.0]]])
    time = torch.tensor([0.25])
    time_next = torch.tensor([0.5])

    target = consistency_velocity_target(x_t, x_t_next, velocity_next, time, time_next)

    # Teacher endpoint: 3 + (1 - 0.5) * 4 = 5; (5 - 2) / (1 - 0.25) = 4.
    assert torch.allclose(target, torch.tensor([[[4.0]]]))


def test_consistency_ema_decay_warms_up_and_is_capped():
    assert consistency_ema_decay(step=0, power=0.75, max_decay=0.9999) == 0.0
    assert 0.0 < consistency_ema_decay(step=10, power=0.75, max_decay=0.9999) < 0.9999
    assert consistency_ema_decay(step=10**12, power=0.75, max_decay=0.99) == 0.99


def test_consistency_config_requires_rectified_flow():
    with pytest.raises(ValueError, match="rectified_flow"):
        SmolVLAConfig(flow_objective="flow_matching", flow_consistency_enabled=True)


@pytest.mark.parametrize(
    "override, expected_message",
    [
        ({"flow_consistency_ratio": 0.0}, "flow_consistency_ratio"),
        ({"flow_consistency_weight": -1.0}, "flow_consistency_weight"),
        ({"flow_consistency_timesteps": 1}, "flow_consistency_timesteps"),
        ({"flow_consistency_ema_power": 0.0}, "flow_consistency_ema_power"),
        ({"flow_consistency_ema_max_decay": 1.0}, "flow_consistency_ema_max_decay"),
    ],
)
def test_invalid_consistency_config_raises(override, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        SmolVLAConfig(flow_consistency_enabled=True, **override)
