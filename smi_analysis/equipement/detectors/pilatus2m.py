import numpy as np
from typing import override, Literal
from pyFAI.detectors import Pilatus2M

class Pilatus2M_SMI(Pilatus2M):
    """
    Pilatus 2M class inherited from the pyFAI Pilatus1M class
    This class is used to add a specific masking for the Pilatus 1M of SMI beamline at BNL
    """

    @override
    def calc_mask(
        self, 
        *,
        bs: tuple[int, int] | None = None,
        bs_kind: Literal["pindiode"] | None = None, 
        optional_mask : Literal["tender"] | None = None
        ):
        """
        :param bs: (string) This is the beamstop position on teh detctor (teh pixels behind will be mask inherently)
        :param bs_kind: (string) What beamstop is in: Only need to be defined if pindiode which have a different shape)
        :param optional_mask: (string) This is usefull for tender x-ray energy and will add extra max at the chips junction
        :return: (a 2D array) A mask array with 0 and 1 with 0s where the image will be masked
        """
        mask = np.logical_not(super().calc_mask())
        mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False

        #Hot pixels
        mask[20, 884], mask[56, 754], mask[111, 620], mask[145, 733], mask[178, 528], mask[
            189, 571] = False, False, False, False, False, False
        mask[372, 462], mask[454, 739], mask[657, 947], mask[869, 544], mask[870, 546], mask[
            870, 547] = False, False, False, False, False, False
        mask[870, 544], mask[871, 545], mask[871, 546], mask[871, 547] = False, False, False, False

        #For tender x-rays
        if optional_mask == 'tender':
            i = 60
            while i < np.shape(mask)[0]:
                if 480 < i < 530:
                    i = 554
                mask[:, i - 2: i + 2] = False
                i += 61

            j = 97
            while j < np.shape(mask)[0]:
                if 150 < j < 250:
                    j = 310
                elif 380 < j < 420:
                    j = 520
                elif 600 < j < 700:
                    j = 734
                elif 790 < j < 890:
                    j = 945
                mask[j - 2: j + 2, :] = False
                j += 100

        #Beamstop
        if bs is not None and bs != [0,0]: # ignore 0,0 for legacy reasons.
            mask[bs[1]:, bs[0] - 11:bs[0] + 11] = False
            if bs_kind == 'pindiode': mask[bs[1]:bs[1] + 40, bs[0] - 22:bs[0] + 22] = False
        return np.logical_not(mask)