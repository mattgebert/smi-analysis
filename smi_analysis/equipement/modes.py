from enum import Enum
from typing import Literal

class MeasurementMode(Enum):
    TRANSMISSION = "transmission"
    GRAZING = "grazing"
    # OPLS = "" # TODO: Is this a mode or a 
    # CMS = ""
    
MeasurementModeType = MeasurementMode | Literal["transmission", "grazing"]