from typing import Dict, List

import numpy as np


class PeriodForSimulation:
    def __init__(
        self,
        event_type_event_times_map: Dict[str, np.ndarray],
        event_types_to_predict: List[str],
        event_types_order: List[str],
    ) -> None:
        self._event_type_event_times_map = event_type_event_times_map
        self._event_types_to_predict = event_types_to_predict
        self._event_types_order = sorted(event_types_order)

    @property
    def event_types_order(self) -> List[str]:
        return self._event_types_order.copy()

    def get_ordered_event_times(self) -> List[np.ndarray]:
        return [
            self._event_type_event_times_map[event_type]
            for event_type in self._event_types_order
        ]
    
    def get_ordered_event_times(
        self,
        event_types_order: List[str]
    ) -> List[np.ndarray]:
        return [
            self._event_type_event_times_map[event_type]
            for event_type in event_types_order
        ]
    
    def get_period_from_time(self, time: float) -> 'PeriodForSimulation':
        return PeriodForSimulation(
            {
                event_type: event_times[event_times >= time]
                for event_type, event_times in self._event_type_event_times_map.items()
            },
            self._event_types_to_predict,
            self._event_types_order,
        )
    
    def get_period_to_time(self, time: float) -> 'PeriodForSimulation':
        return PeriodForSimulation(
            {
                event_type: event_times[event_times <= time]
                for event_type, event_times in self._event_type_event_times_map.items()
            },
            self._event_types_to_predict,
            self._event_types_order,
        )
    
    def get_period_between_times(
        self,
        start_time: float,
        end_time: float,
    ) -> 'PeriodForSimulation':
        return PeriodForSimulation(
            {
                event_type: event_times[
                    (event_times >= start_time) & (event_times <= end_time)
                ]
                for event_type, event_times in self._event_type_event_times_map.items()
            },
            self._event_types_to_predict,
            self._event_types_order,
        )
        
    def get_event_type_event_times_map_to_predict(self) -> Dict[str, np.ndarray]:
        return {
            event_type: self._event_type_event_times_map[event_type]
            for event_type in self._event_types_to_predict
        }