"""
The module for detectors used at the SMI Beamline.

These include the SAXS/WAXS endstation:
- Pilatus 900k
- 

And the Liquid Scattering (OPLS) endstation:
- Pilatus 100k
- Pilatus 300kw (300K pixels, wide config)

And the CMS endstation:
- 

""" # TODO: Update module docstring^

from smi_analysis.equipement.detectors.pilatus1m import Pilatus1M_SMI
from smi_analysis.equipement.detectors.pilatus2m import Pilatus2M_SMI
from smi_analysis.equipement.detectors.pilatus100k import Pilatus100k_OPLS
from smi_analysis.equipement.detectors.pilatus300k import Pilatus300k_OPLS
from smi_analysis.equipement.detectors.pilatus300kw import VerticalPilatus300kw
from smi_analysis.equipement.detectors.pilatus800k import Pilatus800k_CMS
from smi_analysis.equipement.detectors.pilatus900k import Pilatus900k, VerticalPilatus900kw
from enum import Enum
from pyFAI.detectors import Detector

class SMI_DetectorType(Enum):
    PILATUS1M = "Pilatus1m"
    PILATUS2M = "Pilatus2m"
    PILATUS100K_OPLS = "Pilatus100k_OPLS"
    PILATUS300KW_VERT = "Pilatus300kw_Vert"
    PILATUS300K_OPLS = "Pilatus300k_OPLS"
    PILATUS800K_CMS = "Pilatus800k_CMS"
    PILATUS900K = "Pilatus900k"
    PILATUS900KW_VERT = "Pilatus900kw_Vert"
    

DETECTORS: dict[SMI_DetectorType, type[Detector]] = {
    SMI_DetectorType.PILATUS1M : Pilatus1M_SMI,
    SMI_DetectorType.PILATUS2M : Pilatus2M_SMI,
    SMI_DetectorType.PILATUS100K_OPLS : Pilatus100k_OPLS,
    SMI_DetectorType.PILATUS300KW_VERT : VerticalPilatus300kw,
    SMI_DetectorType.PILATUS300K_OPLS : Pilatus300k_OPLS,
    SMI_DetectorType.PILATUS800K_CMS : Pilatus800k_CMS,
    SMI_DetectorType.PILATUS900K : VerticalPilatus900kw,
    SMI_DetectorType.PILATUS900K : Pilatus900k,
    
}

__all__ = [
    # Variables
    "SMI_DetectorType",
    "DETECTORS",
    # Detectors
    "Pilatus1M_SMI",
    "Pilatus2M_SMI",
    "Pilatus100k_OPLS",
    "Pilatus300k_OPLS",
    "VerticalPilatus300kw",
    "Pilatus800k_CMS",
    "Pilatus900k",
    "VerticalPilatus900kw",
]