import numpy as np
from typing import override, Literal
from pyFAI.detectors import Pilatus300kw

class VerticalPilatus300kw(Pilatus300kw):
    """
    VerticalPilatus300kw class inherited from the pyFAI Pilatus300kw class but rotated by 90 deg to fit the position of the WAXS detector at SMI
    This class is used to add a specific masking for the Pilatus 300KW of SMI beamline at BNL
    """

    MAX_SHAPE = (1475, 195)
    MODULE_SIZE = (487, 195)
    MODULE_GAP = (7, 17)
    aliases = ["Pilatus 300kw (Vertical)"]

    @override
    def calc_mask(
        self, 
        *,
        bs: tuple[int, int] | None = None,
        optional_mask : Literal["tender"] | None = None
        ):
        """
        :param bs: (string) This is the beamstop position on teh detctor (teh pixels behind will be mask inherently)
        :param bs_kind: (string) Not used for now but can be used if different beamstop are used
        :param optional_mask: (string) This is usefull for tender x-ray energy and will add extra max at the chips junction
        :return: (a 2D array) A mask array with 0 and 1 with 0s where the image will be masked
        """
        # mask = np.rot90(np.logical_not(detectors.Pilatus300kw().calc_mask()), 1)
        mask = np.rot90(np.logical_not(super().calc_mask()), 1)


        #border of the detector and chips
        mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False
        mask[486, :], mask[494, :], mask[980, :], mask[988, :]= False, False, False, False

        #Dead pixel
        dead_pix_x = [1386, 1387, 1386, 1387, 228, 307, 733, 733, 792, 1211, 1211, 1231, 1232, 1276, 1321, 1366, 1405, 1467, 1355, 1372, 1356]
        dead_pix_y = [96, 96, 97, 97, 21, 67, 170, 171, 37, 109, 110, 74, 74, 57, 81, 181, 46, 188, 85, 89, 106]
        for d_x, d_y in zip(dead_pix_x, dead_pix_y):
            mask[d_x, d_y] = False

        #Hot pixels
        mask[1314, 81] = False
        mask[732, 7], mask[732, 8], mask[733, 8], mask[733, 7], mask[733, 9] = False, False, False, False, False
        mask[1314, 82], mask[1315, 81] = False, False

        mask[674, 133], mask[674, 134], mask[1130, 20], mask[1239, 50] = False, False, False, False

        # For tender x-rays
        if optional_mask == 'tender':
            mask[:, 92:102] = False
            i = 59
            while i < np.shape(mask)[0]:
                if 450 < i < 550:
                    i = 553
                elif 970 < i < 1000:
                    i = 1047
                mask[1475 - i - 6:1475 - i, :] = False
                i += 61

        #Beamstop
        if bs is not None and bs != [0,0]: # ignore 0,0 for legacy reasons.
            mask[bs[1]:, bs[0] - 8 : bs[0] + 8] = False
        return np.logical_not(mask)