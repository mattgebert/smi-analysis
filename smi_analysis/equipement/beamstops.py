"""
Defines the beamstops used in the SMI beamline.
"""
from pyFAI.detectors import Detector
import numpy as np

class beamstop:
    """
    Represents a beamstop used in the SMI beamline.
    
    Parameters
    ----------
    x : float
        The x position of the beamstop in pixels.
    y : float
        The y position of the beamstop in pixels.
    """
    width: float
    """The width of the beamstop in millimeters."""
    height: float
    """The height of the beamstop in millimeters."""
    
    def __init__(self, x, y) -> None:
        self.x = x
        """The x position of the pin-diode beamstop in pixels."""
        self.y = y
        """The y position of the pin-diode beamstop in pixels."""
    
    def width_pixels(self, detector: Detector) -> int:
        """
        The beamstop width in pixels.
        
        Parameters
        ----------
        detector : pyFAI.detectors.Detector
            An instance of the pyFAI detector class, which defines a pixel size.
        """
        return int(np.ceil(detector.pixel1 * self.width))
    
    def heigh_pixels(self, detector: Detector) -> int:
        """
        The beamstop height in pixels.
        
        Parameters
        ----------
        detector : pyFAI.detectors.Detector
            An instance of the pyFAI detector class, which defines a pixel size.
        """
        return int(np.ceil(detector.pixel2 * self.height))
    
    def 
    
class smi_pindiode(beamstop):
    """
    Represents a pin-diode beamstop used in the SMI beamline.
    
    """
    width = 1.0
    """The width of the pin-diode beamstop in detector pixels."""
    height = 40.0
    """The height of the pin-diode beamstop in detector pixels."""
        