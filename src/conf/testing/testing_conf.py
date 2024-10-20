from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TestingConf:
    pair: str
    base_imbalance_level: int
    seconds_simulation_period: int
    seconds_warm_up_period: int

    @classmethod
    def from_dict(cls, conf: Dict[str, Any]) -> "TestingConf":
        return cls(
            pair=conf["pair"],
            base_imbalance_level=conf["base_imbalance_level"],
            seconds_simulation_period=conf["seconds_simulation_period"],
            seconds_warm_up_period=conf["seconds_warm_up_period"],
        )