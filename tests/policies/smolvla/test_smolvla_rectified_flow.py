import pytest
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import flow_training_path, sample_flow_time


def test_smolvla_config_defaults_to_rectified_flow():
    cfg = SmolVLAConfig()

    assert cfg.flow_objective == "rectified_flow"
    assert cfg.flow_time_sampling == "uniform"


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
