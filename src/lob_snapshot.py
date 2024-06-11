from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class LOBSnapshot:
    # Class variables
    PRICE_INDEX: ClassVar[int] = 0
    SIZE_INDEX: ClassVar[int] = 1

    timestamp: int
    bids_from_best: List[Tuple[Decimal, Decimal]]
    asks_from_best: List[Tuple[Decimal, Decimal]]


class LOBSnapshotFactory:
    TIMESTAMP_COLUMN = "timestamp"

    ASK_PRICE_COLUMN_PREFIX = "ask_price_"
    ASK_SIZE_COLUMN_PREFIX = "ask_size_"
    BID_PRICE_COLUMN_PREFIX = "bid_price_"
    BID_SIZE_COLUMN_PREFIX = "bid_size_"

    COLUMN_PREFIXES = [
        ASK_PRICE_COLUMN_PREFIX,
        ASK_SIZE_COLUMN_PREFIX,
        BID_PRICE_COLUMN_PREFIX,
        BID_SIZE_COLUMN_PREFIX,
    ]

    def __init__(self, lob_dataframe: pd.DataFrame, num_levels_in_a_side: int):
        """
        Class that creates LOBSnapshot for each row of the lob_dataframe. The lob_dataframe
        needs to contains these columns: timestamp, ask_price_1, ask_size_1, bid_price_1,
        bid_size_1, ...
        """
        self._lob_dataframe = lob_dataframe
        self._num_levels_in_a_side = num_levels_in_a_side

        if not self._are_column_names_valid():
            raise Exception("Column names not valid!")

    def _are_column_names_valid(self) -> bool:
        column_names = self._lob_dataframe.columns

        if not self.TIMESTAMP_COLUMN in column_names:
            return False

        for i in range(1, self._num_levels_in_a_side + 1):
            for col_prefix in self.COLUMN_PREFIXES:
                if not (col_prefix + str(i)) in column_names:
                    return False

        return True

    def get_lob_snapshots_iterator(self) -> Iterator[LOBSnapshot]:
        for i, row in self._lob_dataframe.iterrows():
            yield self.get_lob_snapshot_from_lob_dataframe_row(row)

    def get_lob_snapshot_from_lob_dataframe_row(self, row: pd.Series) -> LOBSnapshot:
        timestamp = int(row[self.TIMESTAMP_COLUMN])

        bid_row = row[
            (row.index.str.startswith(self.BID_PRICE_COLUMN_PREFIX))
            | (row.index.str.startswith(self.BID_SIZE_COLUMN_PREFIX))
        ]
        bid_levels = self._get_side_orderbook_levels(bid_row, True)

        ask_row = row[
            (row.index.str.startswith(self.ASK_PRICE_COLUMN_PREFIX))
            | (row.index.str.startswith(self.ASK_SIZE_COLUMN_PREFIX))
        ]
        ask_levels = self._get_side_orderbook_levels(ask_row, False)

        return LOBSnapshot(timestamp, bid_levels, ask_levels)

    def _get_side_orderbook_levels(
        self, row: pd.Series, is_bid: True
    ) -> List[Tuple[Decimal, Decimal]]:
        orderbook_levels = []

        price_column_prefix = (
            self.BID_PRICE_COLUMN_PREFIX if is_bid else self.ASK_PRICE_COLUMN_PREFIX
        )
        size_column_prefix = (
            self.BID_SIZE_COLUMN_PREFIX if is_bid else self.ASK_SIZE_COLUMN_PREFIX
        )

        for i in range(1, self._num_levels_in_a_side + 1):
            price = Decimal(row[price_column_prefix + str(i)])
            size = Decimal(row[size_column_prefix + str(i)])
            orderbook_levels.append((price, size))

        return orderbook_levels
