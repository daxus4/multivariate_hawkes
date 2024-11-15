from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CoeConf:
    pairs: List[str]
    bi_levels: List[int]
    training_time_seconds: List[int]
    simulation_periods_seconds: List[int]

    @classmethod
    def from_dict(cls, conf: Dict[str, Any]) -> "CoeConf":
        return cls(
            pairs=conf["pairs"],
            bi_levels=conf["bi_levels"],
            training_time_seconds=conf["training_time_seconds"],
            simulation_periods_seconds=conf["simulation_periods_seconds"],
        )
    