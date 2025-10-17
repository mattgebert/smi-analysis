import numpy as np
from pyFAI.detectors import Pilatus

class Pilatus800k_CMS(Pilatus):
    """
    Pilatus 800k detector, assembly of 3x3 modules
    Available at NSLS-II 11-BM.

    This is different from the "Pilatus CdTe 900kw" available at ESRF ID06-LVP which is 1x9 modules
    """
    MAX_SHAPE = (1043, 981)
    aliases = ["Pilatus 800k cms"]
    
    def calc_mask(
        self, 
        *,
        bs: tuple[int, int] | None = None,
        ):
        """
        :param bs: (string) This is the beamstop position on teh detctor (teh pixels behind will be mask inherently)
        :return: (a 2D array) A mask array with 0 and 1 with 0s where the image will be masked
        """
        mask = np.logical_not(np.zeros(self.MAX_SHAPE))
        mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False
        mask[:, 486:494]= False
        
        #The two bottom missing modules
        mask[620:, :486]= False

        #Beamstop
        if bs is not None and bs != [0, 0]:
            mask[bs[1]:, bs[0] - 8:bs[0] + 8] = False
        return np.logical_not(mask)