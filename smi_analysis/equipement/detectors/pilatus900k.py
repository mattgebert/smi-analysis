"""
Module for the Pilatus 900k detector, assembly of 3 vertically aligned sets of 3 modules.

The two outer modules are tilted towards the sample.
"""
from typing import override, Literal
import numpy as np
from pyFAI.detectors import Pilatus

class Pilatus900k(Pilatus):
    """
    Pilatus 900k detector, assembly of 3x3 modules.
    
    Available at NSLS-II 12-ID.
    This is different from the "Pilatus CdTe 900kw" available at ESRF ID06-LVP which is 1x9 modules.
    """
    MAX_SHAPE = (619, 1475)
    aliases = ["Pilatus 900k"]
    
    PILATUS900KW_CORRECTION_GRADIENT: float = -0.3/20
    """
    The angular correction coefficient required to adjust a WAXS-arm angle of any magnitude.
    
    I.e. True Angle = Measured Angle * (1 + PILATUS900KW_CORRECTION_GRADIENT) + PILATUS900KW_CORRECTION_OFFSET
    """
    
    PILATUS900KW_CORRECTION_OFFSET: float = -0.06
    """
    The correction required (in degrees) to adjust a WAXS-arm at zero degrees.
    
    I.e. True Angle = Measured Angle * (1 + PILATUS900KW_CORRECTION_GRADIENT) + PILATUS900KW_CORRECTION_OFFSET
    """
    
    PILATUS900KW_PANEL_ANGLE: float = 7.47
    """The PILATUS900KW detector panel angles (in degrees) for each image."""
    

class VerticalPilatus900kw(Pilatus900k):
    """
    VerticalPilatus900kw class inherited from the pyFAI Pilatus300k class but rotated by 90 deg to fit the position of the WAXS detector at SMI
    This class is used to add a specific masking for the Pilatus 900KW of SMI beamline at BNL
    """

    MAX_SHAPE = (1475, 195)
    MODULE_SIZE = (487, 195)

    aliases = ["Pilatus 900kw (Vertical)"]

    @override
    def calc_mask(
        self, 
        *,
        bs: tuple[int, int] | None = None,
        optional_mask : Literal["tender"] | None = None
        ):
        """
        :param bs: (string) This is the beamstop position on teh detector (teh pixels behind will be mask inherently)
        :param bs_kind: (string) Not used for now but can be used if different beamstop are used
        :param optional_mask: (string) This is useful for tender x-ray energy and will add extra max at the chips junction
        :return: (a 2D array) A mask array with 0 and 1 with 0s where the image will be masked
        """
        # mask = np.rot90(np.logical_not(Pilatus900k().calc_mask()), 1)
        mask = np.rot90(np.logical_not(super().calc_mask()), 1)

        # Border of detector
        mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False

        # Hot pixels
        mask[15:19, 281:285] = False
        mask[305:309, 566:570] = False
        mask[292:296, 571:575] = False
        mask[305:309, 1108:1113] = False
        mask[401:403, 579:581] = False
        mask[182:184, 259:261] = False
        mask[19:21, 287:289] = False
        mask[1291:1294, 303:305] = False
        mask[1254, 259] = False

        if optional_mask == 'tender':
            #vertical gaps of pilatus for each module
            mask[:, 92:102] = False
            mask[:, 304:314] = False
            mask[:, 516:526] = False

            #horizontal gaps of pilatus for each module
            i = 59
            while i < np.shape(mask)[0]:
                if 450 < i < 550:
                    i = 553
                elif 970 < i < 1000:
                    i = 1047
                mask[1475 - i - 6:1475 - i, :] = False
                i += 61

        #Beamstop
        if bs is not None:
            mask[bs[1]:, bs[0] - 8: bs[0] + 8] = False
    
        mask = np.logical_not(mask)
        return mask