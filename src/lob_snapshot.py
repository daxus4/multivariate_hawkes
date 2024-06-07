from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class LOBSnapshot:
    # Class variables
    PRICE_INDEX: int = 0
    SIZE_INDEX: int = 1

    timestamp_ms: int
    bids_from_best: List[Tuple[float, float]]
    asks_from_best: List[Tuple[float, float]]
