from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TrainingConf:
    pair: str
    seconds_in_a_period: int

    @classmethod
    def from_dict(cls, conf: Dict[str, Any]) -> "TrainingConf":
        return cls(
            pair=conf["pair"],
            seconds_in_a_period=conf["seconds_in_a_period"],
        )
    