"""
The SMI Analysis equipement module.

Provides class definitions for characteristics of the beamline.
"""
# TODO: Support beamstops, and cleanup beamstop masking in detectors.

from smi_analysis.equipement.modes import MeasurementMode, MeasurementModeType
from smi_analysis.equipement.detectors import SMI_DetectorType, DETECTORS