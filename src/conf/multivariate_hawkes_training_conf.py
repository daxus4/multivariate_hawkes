from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MultivariateHawkesTrainingConf:
    num_levels_for_which_save_events: int
    num_levels_in_a_side: int
    period_df_path: str
    seconds_in_a_period: int
    lob_df_folder_path: str
    lob_df_file_prefix: str
    base_imbalance_level: int

    @classmethod
    def from_dict(cls, conf: Dict[str, Any]) -> "MultivariateHawkesTrainingConf":
        return cls(
            num_levels_for_which_save_events=conf["num_levels_for_which_save_events"],
            num_levels_in_a_side=conf["num_levels_in_a_side"],
            period_df_path=conf["period_df_path"],
            seconds_in_a_period=conf["seconds_in_a_period"],
            lob_df_folder_path=conf["lob_df_folder_path"],
            lob_df_file_prefix=conf["lob_df_file_prefix"],
            base_imbalance_level=conf["base_imbalance_level"],
        )