from decimal import Decimal
from enum import Enum
from typing import List, Tuple

from lob_snapshot import LOBSnapshot


class LOBEventType(Enum):
    # CHANGER means that the order changed the best bid/ask price
    # REMOVE_LIQUIDITY means that the order is a market order or is cancelled

    BID_REMOVE_LIQUIDITY_ORDER_CHANGER = 1
    ASK_REMOVE_LIQUIDITY_ORDER_CHANGER = 2
    BID_REMOVE_LIQUIDITY_ORDER_NOT_CHANGER = 3
    ASK_REMOVE_LIQUIDITY_ORDER_NOT_CHANGER = 4
    BID_LIMIT_ORDER_CHANGER = 5
    ASK_LIMIT_ORDER_CHANGER = 6
    BID_LIMIT_ORDER_NOT_CHANGER = 7
    ASK_LIMIT_ORDER_NOT_CHANGER = 8
    BID_MARKET_ORDER_CHANGER = 9
    ASK_MARKET_ORDER_CHANGER = 10
    BID_MARKET_ORDER_NOT_CHANGER = 11
    ASK_MARKET_ORDER_NOT_CHANGER = 12
    BID_CANCELLED_ORDER_CHANGER = 13
    ASK_CANCELLED_ORDER_CHANGER = 14
    BID_CANCELLED_ORDER_NOT_CHANGER = 15
    ASK_CANCELLED_ORDER_NOT_CHANGER = 16


class LOBEventTypeReconstructor:
    def get_happened_event_types(
        self, prev_lob_snapshot: LOBSnapshot, curr_lob_snapshot: LOBSnapshot
    ) -> List[LOBEventType]:
        happened_event_types = []

    def _has_best_price_changed(
        self,
        prev_best_price: Decimal,
        curr_best_price: Decimal,
    ) -> bool:
        return prev_best_price != curr_best_price

    def _is_curr_best_price_more_aggressive(
        self,
        prev_best_price: Decimal,
        curr_best_price: Decimal,
        is_bid_side: bool,
    ) -> bool:
        if is_bid_side:
            return curr_best_price > prev_best_price
        else:
            return curr_best_price < prev_best_price
