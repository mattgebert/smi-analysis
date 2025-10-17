import numpy as np, numpy.typing as npt
from typing import override, Literal
from pyFAI.detectors import Detector

class Rayonix(Detector):
    """
    Rayonix detector: generic description containing mask algorithm

    Nota: 1920x1920 pixels, 0.109mm pixel size
    """
    
    MAX_SHAPE = (1920, 1920)
    aliases = ["rayonix"]

    def __init__(self, pixel1=109e-6, pixel2=109e-6, max_shape=None):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2, max_shape=max_shape)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
               (self.name, self._pixel1, self._pixel2)
               
    @override
    def calc_mask( # type: ignore
        self, 
        *,
        img : npt.ArrayLike | None = None, 
        threshold: float | int = 15,
    ):
        if img is None:
            mask = True
        else:
            img = np.asarray(img)
            mask = np.ones_like(img, dtype=bool)
            mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False
            mask[np.where(img < threshold)] = False
            
        return np.logical_not(mask)