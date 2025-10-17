"""
`SMI Analysis` is a toolkit to reduce and analyse data taken at the NSLS2, 12-ID Beamline, NY. 
"""
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# Define accesible properties at the root level.
from smi_analysis.equipement.modes import MeasurementMode, MeasurementModeType
from smi_analysis.equipement.detectors import DETECTORS, SMI_DetectorType
from smi_analysis.SMI_beamline import SMI_ExperimentConfig