from dataclasses import dataclass, field

from torch import nn

from werewolf.models.risk.twd_risk_layer import TWDRiskLayer
from werewolf.models.twd_tom.backbone import ToMBackbone, ToMBackboneConfig


@dataclass
class TWDToMConfig:
    tom_config: ToMBackboneConfig = field(
        default_factory=ToMBackboneConfig
    )
    twd_tau: float = 0.7


class TWDToMModel(nn.Module):
    def __init__(
        self,
        config: TWDToMConfig | None = None,
        tom_backbone: nn.Module | None = None,
        twd_layer: nn.Module | None = None,
    ):
        super().__init__()
        self.config = TWDToMConfig() if config is None else config
        self.tom_backbone = (
            ToMBackbone(self.config.tom_config)
            if tom_backbone is None
            else tom_backbone
        )
        self.twd_layer = (
            TWDRiskLayer(tau=self.config.twd_tau)
            if twd_layer is None
            else twd_layer
        )

    def forward(
        self,
        event_tokens,
        attention_mask=None,
        context=None,
        observer_id=None,
    ):
        tom_kwargs = {"attention_mask": attention_mask}
        if observer_id is not None:
            tom_kwargs["observer_id"] = observer_id
        tom_outputs = self.tom_backbone(event_tokens, **tom_kwargs)
        twd_outputs = self.twd_layer(
            tom_outputs["wolf_prob"],
            context=context,
        )
        return {
            "hidden_states": tom_outputs["hidden_states"],
            "wolf_logits": tom_outputs["wolf_logits"],
            "wolf_prob": tom_outputs["wolf_prob"],
            "region_probs": twd_outputs["region_probs"],
            "risks": twd_outputs["risks"],
            "hard_region": twd_outputs["hard_region"],
            "costs": twd_outputs["costs"],
        }
