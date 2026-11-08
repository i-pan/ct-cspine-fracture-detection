import pytest
import torch

from skp.configs import Config
from skp.models.classification import mil_3d


class TinyBackbone(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def _cfg(**overrides):
    cfg = Config(
        backbone="tiny",
        pretrained=False,
        num_input_channels=2,
        image_height=8,
        image_width=8,
        num_classes=3,
        pool="avg",
        pool_params=None,
        dropout=0.0,
        normalization="linear",
        normalization_params={
            "input_min": 0.0,
            "input_max": 1.0,
            "output_min": 0.0,
            "output_max": 1.0,
        },
        enable_gradient_checkpointing=False,
        load_pretrained_backbone=None,
        load_pretrained_model=None,
        freeze_backbone=False,
        mil_pool_temperature=1.0,
        mil_pool_normalize=True,
    )
    cfg.__dict__.update(overrides)
    return cfg


def test_log_mean_exp_is_invariant_to_repeated_slices():
    one_slice = torch.tensor([[[2.0, -1.0]]])
    repeated = one_slice.repeat(1, 5, 1)

    one_pooled = mil_3d.log_sum_exp_pool(one_slice, normalize=True)
    repeated_pooled = mil_3d.log_sum_exp_pool(repeated, normalize=True)
    raw_pooled = mil_3d.log_sum_exp_pool(repeated, normalize=False)

    assert torch.allclose(one_pooled, repeated_pooled)
    assert torch.allclose(raw_pooled, repeated_pooled + torch.log(torch.tensor(5.0)))


def test_log_sum_exp_pool_ignores_masked_slices():
    slice_logits = torch.tensor([[[1.0], [1000.0], [3.0]]])
    mask = torch.tensor([[True, False, True]])

    pooled = mil_3d.log_sum_exp_pool(slice_logits, mask=mask, normalize=True)
    expected = torch.logsumexp(torch.tensor([1.0, 3.0]), dim=0) - torch.log(
        torch.tensor(2.0)
    )

    assert pooled.item() == pytest.approx(expected.item())
    with pytest.raises(ValueError, match="at least one valid slice"):
        mil_3d.log_sum_exp_pool(slice_logits, mask=torch.zeros_like(mask))


def test_log_mean_exp_uses_each_samples_valid_slice_count():
    slice_logits = torch.tensor(
        [
            [[2.0], [2.0], [1000.0], [1000.0]],
            [[2.0], [2.0], [2.0], [2.0]],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    pooled = mil_3d.log_sum_exp_pool(
        slice_logits,
        mask=mask,
        normalize=True,
    )
    assert torch.allclose(pooled, torch.full_like(pooled, 2.0))


def test_mil_model_returns_volume_and_slice_outputs(monkeypatch):
    monkeypatch.setattr(
        mil_3d,
        "create_model",
        lambda *args, **kwargs: TinyBackbone(kwargs["in_chans"]),
    )
    model = mil_3d.Net(_cfg())
    batch = {
        "x": torch.rand(2, 2, 5, 8, 8),
        "mask": torch.tensor(
            [
                [True, True, True, True, True],
                [True, True, True, False, False],
            ]
        ),
    }

    out = model(batch, return_features=True)

    assert out["logits"].shape == (2, 3)
    assert out["slice_logits"].shape == (2, 5, 3)
    assert out["slice_features"].shape == (2, 5, 4)
    assert torch.allclose(
        out["logits"],
        mil_3d.log_sum_exp_pool(
            out["slice_logits"],
            mask=batch["mask"],
            normalize=True,
        ),
    )


def test_mil_model_uses_standard_criterion_contract(monkeypatch):
    monkeypatch.setattr(
        mil_3d,
        "create_model",
        lambda *args, **kwargs: TinyBackbone(kwargs["in_chans"]),
    )
    model = mil_3d.Net(_cfg(num_classes=1))

    class Criterion(torch.nn.Module):
        def forward(self, out, batch):
            return {"loss": (out["logits"] - batch["y"]).square().mean()}

    model.set_criterion(Criterion())
    out = model(
        {
            "x": torch.rand(2, 2, 4, 8, 8),
            "y": torch.rand(2, 1),
        },
        return_loss=True,
    )

    out["loss"].backward()
    assert out["loss"].ndim == 0
    assert model.linear.weight.grad is not None


def test_mil_model_constructs_with_timm_backbone():
    model = mil_3d.Net(
        _cfg(
            backbone="resnet18",
            num_input_channels=3,
            image_height=32,
            image_width=32,
            num_classes=2,
        )
    )

    out = model({"x": torch.rand(1, 3, 2, 32, 32)})

    assert out["logits"].shape == (1, 2)
    assert out["slice_logits"].shape == (1, 2, 2)
