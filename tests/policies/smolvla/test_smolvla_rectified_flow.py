import pytest
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    flow_solver_step,
    flow_training_path,
    probeflow_euler_sample,
    probeflow_linearity,
    probeflow_schedule_steps,
    sample_flow_time,
)


def test_smolvla_config_defaults_to_rectified_flow():
    cfg = SmolVLAConfig()

    assert cfg.flow_objective == "rectified_flow"
    assert cfg.flow_time_sampling == "uniform"
    assert cfg.flow_solver == "euler"
    assert cfg.probeflow_probe_time == 0.5
    assert cfg.probeflow_max_steps == 4


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


def test_probeflow_linearity_ignores_unexecuted_and_padded_coordinates():
    v_start = torch.ones(1, 3, 3)
    v_probe = v_start.clone()
    v_probe[:, 1:, :] = -1
    v_probe[:, :, 1:] = -1

    similarity = probeflow_linearity(v_start, v_probe, action_horizon=1, action_dim=1)

    assert torch.allclose(similarity, torch.ones(1))


def test_probeflow_schedule_maps_linear_paths_to_two_steps_and_curved_paths_to_four():
    similarity = torch.tensor([1.0, 0.996, 0.99, -1.0])

    steps = probeflow_schedule_steps(
        similarity,
        epsilon=0.008,
        max_steps=4,
        step_increment=2,
    )

    assert torch.equal(steps, torch.tensor([2, 2, 4, 4]))


def test_probeflow_linear_path_reuses_two_probe_evaluations():
    noise = torch.zeros(1, 2, 1)
    calls = []

    def constant_velocity(x_t, time):
        calls.append(time)
        return torch.full_like(x_t, 2.0)

    sample, similarity, steps, nfe = probeflow_euler_sample(
        noise=noise,
        predict_velocity=constant_velocity,
        action_horizon=2,
        action_dim=1,
        probe_time=0.5,
        epsilon=0.008,
        max_steps=4,
        step_increment=2,
    )

    assert torch.allclose(sample, torch.full_like(noise, 2.0))
    assert torch.allclose(similarity, torch.ones(1))
    assert steps == 2
    assert nfe == 2
    assert calls == [0.0, 0.5]


def test_probeflow_dense_path_counts_discarded_probe_evaluation():
    noise = torch.zeros(1, 1, 1)
    calls = []

    def changing_velocity(x_t, time):
        calls.append(time)
        return torch.ones_like(x_t) if time == 0.0 else -torch.ones_like(x_t)

    _, similarity, steps, nfe = probeflow_euler_sample(
        noise=noise,
        predict_velocity=changing_velocity,
        action_horizon=1,
        action_dim=1,
        probe_time=0.5,
        epsilon=0.008,
        max_steps=4,
        step_increment=2,
    )

    assert torch.allclose(similarity, -torch.ones(1))
    assert steps == 4
    assert nfe == 5
    assert calls == [0.0, 0.5, 0.25, 0.5, 0.75]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"probeflow_probe_time": 0.0}, "probeflow_probe_time"),
        ({"probeflow_epsilon": 0.0}, "probeflow_epsilon"),
        ({"probeflow_max_steps": 5}, "reachable"),
        ({"probeflow_log_every": -1}, "probeflow_log_every"),
        ({"flow_solver": "probe_euler", "flow_objective": "flow_matching"}, "rectified_flow"),
        ({"flow_solver": "probe_euler", "compile_model": True}, "compile_model"),
    ],
)
def test_invalid_probeflow_config_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SmolVLAConfig(**kwargs)
