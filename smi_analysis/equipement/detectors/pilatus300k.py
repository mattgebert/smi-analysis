import numpy as np
from typing import override, Literal
from pyFAI.detectors import Pilatus300k

class Pilatus300k_OPLS(Pilatus300k):
    """
    Pilatus 100k class inherited from the pyFAI Pilatus1M class
    This class is used to add a specific masking for the Pilatus 100k of OPLS beamline at BNL
    """
    MAX_SHAPE = (619, 487)

    @override
    def calc_mask(
        self, 
        *,
        bs: tuple[int, int] | None = None
        ):
        """
        :param bs: (string) This is the beamstop position on teh detctor (teh pixels behind will be mask inherently)
        :param bs_kind: (string) What beamstop is in: Only need to be defined if pindiode which have a different shape)
        :param optional_mask: (string) This is usefull for tender x-ray energy and will add extra max at the chips junction
        :return: (a 2D array) A mask array with 0 and 1 with 0s where the image will be masked
        """
        mask = np.logical_not(np.zeros(self.MAX_SHAPE))
        mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False

        #Hot pixels needs to be defines
        # mask[20, 884], mask[56, 754], mask[111, 620], mask[145, 733], mask[178, 528], mask[
        #     189, 571] = False, False, False, False, False, False

        #Beamstop
        if bs is not None and bs != [0,0]: # ignore 0,0 for legacy reasons.
            mask[bs[1]:, bs[0] - 8:bs[0] + 8] = False
        return np.logical_not(mask)
        
