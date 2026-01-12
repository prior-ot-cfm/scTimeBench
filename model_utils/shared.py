"""
Definitions to be shared by the benchmark and the model implementations.
"""
from enum import Enum


class ObservationColumns(Enum):
    CELL_TYPE = "cell_type"
    TIMEPOINT = "timepoint"
