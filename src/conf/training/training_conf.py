from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TrainingConf:
    pair: str
    base_imbalance_level: int
    seconds_in_a_period: int

    @classmethod
    def from_dict(cls, conf: Dict[str, Any]) -> "TrainingConf":
        return cls(
            pair=conf["pair"],
            base_imbalance_level=conf["base_imbalance_level"],
            seconds_in_a_period=conf["seconds_in_a_period"],
        )
    