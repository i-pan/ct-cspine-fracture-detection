from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import lightning as L
import torch


class HaltonPruningCallback(L.Callback):
    """Conservatively prune sequential sweep trials against matched epochs."""

    def __init__(
        self,
        *,
        state_path: str | Path,
        trial_id: str,
        margins: dict[int | str, float],
        metric: str = "val_metric",
        mode: str = "max",
    ) -> None:
        super().__init__()
        if mode != "max":
            raise ValueError("Halton pruning currently supports mode='max' only.")
        self.state_path = Path(state_path)
        self.trial_id = trial_id
        self.margins = {int(epoch): float(value) for epoch, value in margins.items()}
        self.metric = metric

    def _read_state(self) -> dict:
        if not self.state_path.is_file():
            return {"best_by_epoch": {}, "observations": []}
        return json.loads(self.state_path.read_text())

    def _write_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_path, self.state_path)

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        del pl_module
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch) + 1
        value = trainer.callback_metrics.get(self.metric)
        if value is None:
            return
        metric_value = float(value.detach().cpu() if torch.is_tensor(value) else value)

        should_prune = False
        incumbent = None
        if trainer.is_global_zero:
            state = self._read_state()
            incumbent_value = state.get("best_by_epoch", {}).get(str(epoch))
            incumbent = None if incumbent_value is None else float(incumbent_value)
            margin = self.margins.get(epoch)
            should_prune = (
                margin is not None
                and incumbent is not None
                and metric_value < incumbent - margin
            )
            state.setdefault("observations", []).append(
                {
                    "trial_id": self.trial_id,
                    "epoch": epoch,
                    "metric": metric_value,
                    "incumbent_before_update": incumbent,
                    "pruned": should_prune,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            best_by_epoch = state.setdefault("best_by_epoch", {})
            if incumbent is None or metric_value > incumbent:
                best_by_epoch[str(epoch)] = metric_value
                state.setdefault("best_trial_by_epoch", {})[str(epoch)] = self.trial_id
            self._write_state(state)

        decision = torch.tensor(
            int(should_prune),
            device=trainer.strategy.root_device,
            dtype=torch.int8,
        )
        decision = trainer.strategy.broadcast(decision, src=0)
        should_prune = bool(decision.item())
        if should_prune:
            trainer.should_stop = True
            if trainer.is_global_zero:
                margin = self.margins[epoch]
                print(
                    f"Halton pruning {self.trial_id} after epoch {epoch}: "
                    f"{self.metric}={metric_value:.4f}, incumbent={incumbent:.4f}, "
                    f"margin={margin:.4f}."
                )
