import enum
from pyFAI import azimuthalIntegrator
from pygix import Transform
from smi_analysis import Detector, stitch, integrate1D
import os
import fabio
import numpy as np
import copy
import datetime
from typing import Literal
import numpy.typing as npt
import warnings

class SMI_geometry():
    """
    The SMI_geometry class is a class that contains all the information about the geometry of the beamline.
    
    Used for calculating and stiching images together from various detector positions/angles.
    
    Parameters
    ----------
    geometry : Literal['Transmission'] | Literal['Reflection']
        The measurement geometry.
    sdd : float
        Sample to detector distance in millimeters.
    wav : float
        The wavelength of the X-ray beam in meters.
    center : tuple[int|float, int|float]
        Coordinates of the beam centre at 0 degrees.
    bs_pos : list[tuple[int, int]]
        The position of the center of the beam stop for each dselfetector angle; [0,0] implies not measured.
    detector : Literal['Pilatus900kw'] | Literal['Pilatus1m']
        Type of detector.
    det_ini_angle : float
        The initial angle of the detector.
    det_angle_step : float
        The step between each detector angle.
    det_angles : list[int | float] | npt.NDArray[np.float64 | np.int_]
        The angles of the detector in radians.
    alphai : float
        The angle of incidence of the X-ray beam in degrees.
    bs_kind : Literal["pindiode"] | Literal['rod'] | None
        The type of beamstop used for the measurement.
    """
    
    def __init__(self,
                 geometry: Literal['Transmission'] | Literal['Reflection'],
                 sdd: float,
                 wav: float,
                 center: tuple[int|float, int|float],
                 bs_pos: list[tuple[int, int]],
                 detector: Literal['Pilatus900kw'] | Literal['Pilatus1m'] = 'Pilatus900kw',
                 det_ini_angle: float=0,
                 det_angle_step: float=0,
                 det_angles: list[int | float] | npt.NDArray[np.float64 | np.int_]=[],
                 alphai=0,
                 bs_kind=None):

        self.geometry: Literal['Transmission'] | Literal['Reflection'] = geometry
        """The measurement geometry."""
        self.sdd: float = sdd
        """Sample to detector distance in millimeters."""
        self._wav: float = wav
        """The wavelength of the X-ray beam in meters."""
        self._alphai: float = -alphai
        """The angle of incidence of the X-ray beam in degrees."""
        self.center: tuple[int|float, int|float] = center
        self.bs: list[tuple[int, int]] = bs_pos
        self.detector: Literal['Pilatus900kw'] | Literal['Pilatus1m'] = detector

        self._perpendicular_correction: bool = False
        """Attribute to track if data has been corrected for perpendicular geometry in stiching_data method."""

        self.det_ini_angle = det_ini_angle
        self.det_angle_step = det_angle_step
        self._cal_angles: list[float] = self.calibrate_waxs_angles(det_angles)
        """The calibrated detector angles that will be measured
        
        The pilatus 1M detector images are split into three panels, each treated with a different angle.
        """
        self._det_angles: list[float] = det_angles
        """
        The user-entered detector angles that will be measured.
        
        See Also
        --------
        _cal_angles : list
            The list of calibrated angles that will be used by the integrator.
        """

        self.ai: list[azimuthalIntegrator.AzimuthalIntegrator | Transform] = []
        """Azimuthal integrator objects for each detector angle"""
        self.masks = []
        self.cake = []
        self.inpaints, self.mask_inpaints = [], []
        self.img_st, self.mask_st = [], []
        self.bs_kind = bs_kind
        self.scales = 1

        self.define_detector()

        # Initialization of all components of SMI geometry
        self.imgs = []
        self.cake, self.q_cake, self.chi_cake = [], [], []
        self.qp, self.qz = [], []
        self.chi_azi, self.I_azi = [], []
        self.q_hor, self.I_hor = [], []
        self.q_ver, self.I_ver = [], []
        self.q_rad, self.I_rad = [], []
        
    @property
    def wav(self):
        """
        Wavelength of the X-ray beam in meters.
        
        Setting propogates the wavelength change to the azimuthal integrators.
        
        Parameters
        ----------
        value : float
            Wavelength of the X-ray beam in meters.
        """
        return self._wav
    
    @wav.setter
    def wav(self, value):
        if self._wav != value:
            self._wav = value
            
            ## THIS WAS TOO SLOW. I COMMENTED IT OUT.
            # # Reset the azimuthal integrators, which depend on the wavelength
            # self.ai = []
            
            # INSTEAD, UPDATE THE AI OBJECT PROPERTIES
            for ai in self.ai:
                ai.set_wavelength(value) # From pyFai.geometry.core.Geometry
            
    @property
    def alphai(self):
        """
        Angle of incidence of the X-ray beam in degrees.
        
        Setting propogates the incident angle change to the azimuthal integrators.
        Note that when setting alphai, SMI_geometry changes the sign of the angle,
        to match the constructor convention (which also converts from radians to degress).
        As this value will be used internally, we keep the value in degress, and make the setter
        also degrees rather than radians like the constructor.
        
        Parameters
        ----------
        value : float
            Angle of incidence of the X-ray beam in degrees.
        """
        return -self._alphai
    
    @alphai.setter
    def alphai(self, value):
        # Just like initialisation, sign swap the incoming angle of incidence
        v = -value
        if self._alphai != v: # if it changes, save it.
            self._alphai = v # save the negative.
            
            ## THIS WAS TOO SLOW. I COMMENTED IT OUT.
            # # Reset the azimuthal integrators, which depend on the incident angle
            # self.ai = []

            # INSTEAD, UPDATE THE AI OBJECT PROPERTIES
            for ai in self.ai:
                if self.geometry == 'Reflection' and isinstance(ai, Transform):
                    ai.set_incident_angle(self._alphai) # Push the negative.

    PILATUS900KW_CORRECTION_GRADIENT: float = -0.3/20
    """
    The angular correction coefficient required to adjust a WAXS-arm angle of any magnitude.
    """
    
    PILATUS900KW_CORRECTION_OFFSET: float = -0.06
    """The correction required (in degrees) to adjust a WAXS-arm at zero degrees."""
    
    PILATUS900KW_PANEL_ANGLE: float = 7.47
    """The PILATUS900KW detector panel angles (in degrees) for each image."""
    
    def calibrate_waxs_angles(self, angles: list[float]) -> list[float]:
        """
        Prepares angles for use in the SMI detector.

        Adjusts angles for offsets found in the detector, and also converts
        angles in degrees to radians.

        If the detector is `Pilatus900kw` then additional angles for the side-panels of the detector are also calculated.

        Returns
        -------
        list[float]
            Corrections of input WAXS angles in radians.
        """
        calibrated_angles = np.asarray(angles) * (1 + self.PILATUS900KW_CORRECTION_GRADIENT) + self.PILATUS900KW_CORRECTION_OFFSET
        if self.detector is None:
            warnings.warn("The detector has not been defined. Corrected WAXS angles might be incorrect.")
        elif self.detector == "Pilatus900kw":
            # Split each detector angle into 3 for each subpanel.
            new_angles = []
            for angle in calibrated_angles:
                new_angles.append(angle - self.PILATUS900KW_PANEL_ANGLE)
                new_angles.append(angle)
                new_angles.append(angle + self.PILATUS900KW_PANEL_ANGLE)
            calibrated_angles = new_angles
        # Convert to radians
        radian_angles = np.deg2rad(calibrated_angles)
        return radian_angles

    @property
    def det_angles(self) -> list[float]:
        """
        The user-defined angles of the detector to measure.
        
        Parameters
        ----------
        angles : list[float] | float
            A new list of WAXS detector angles.
        
        Returns
        -------
        list[float]
            The list of user-defined angle values.
        
        See Also
        --------
        SMI_geometry._cal_angles
            The corrected angles.
            
        """
        return self._det_angles
    
    @det_angles.setter
    def det_angles(self, angles:list[float] | float):
        if isinstance(angles, float):
            angles = [angles]
        # only update if the angles have changed.
        if self._det_angles != angles:
            self._det_angles = angles
            
            # Calculate the new angles
            self._cal_angles = self.calibrate_waxs_angles(angles)
            
            # Update the azimuthal integrator objects
            if len(self.ai) != 0:
                if len(self.ai) == len(self._cal_angles):
                    for i, (angle, ai) in zip(self._cal_angles, self.ai):
                        if isinstance(ai, azimuthalIntegrator.AzimuthalIntegrator):
                            ai.set_rot1(angle)
                        elif isinstance(ai, Transform):
                            ai.set_rot1(angle)
                        else:
                            warnings.warn(f"The integrator object `{ai}` cannot be updated for a new WAXS detector angle. Resetting all integrators.")
                            self.ai = []
                            break
                else:
                    # Reset the integrators so they will be re-calculated at next stitch.
                    self.ai = []        

    def define_detector(self):
        """
        Definition of the detectors in pyFAI framework, with a default mask
        """
        if self.detector == 'Pilatus1m':
            self.det = Detector.Pilatus1M_SMI()
        elif self.detector == 'Pilatus900kw':
            self.det = Detector.VerticalPilatus900kw()
        elif self.detector == 'Pilatus300kw':
            self.det = Detector.VerticalPilatus300kw()
        elif self.detector == 'rayonix':
            self.det = Detector.Rayonix()
        elif self.detector == 'Pilatus100k_OPLS':
            self.det = Detector.Pilatus100k_OPLS()
        elif self.detector == 'Pilatus300k_OPLS':
            self.det = Detector.Pilatus300k_OPLS()
        elif self.detector == 'Pilatus800k_CMS':
            self.det = Detector.Pilatus800k_CMS()
        else:
            raise Exception('Unknown detector for SMI. Should be either: Pilatus1m or Pilatus300kw or Pilatus900kw or rayonix')

    def open_data(self, path, lst_img, optional_mask=None):
        """
        Function to open the data in a given path and with a name. A list of file needs to be pass for
        stitching data taken at different waxs detector angle.
        :param path: string. Path to the file on your computer
        :param lst_img: list of string. List of filename to load sitting in the path folder
        :param optional_mask: string. Can be 'tender' to mask extra chips of the detectors
        :return:
        """
        if self.detector is None:
            self.define_detector()

        # Reset the images and the masks
        self.imgs = []
        self.masks = []
        self._perpendicular_correction = False
        if len(lst_img) != len(self.bs):
            self.bs = self.bs + [[0, 0]]*(len(lst_img) - len(self.bs))

        for i, (img, bs) in enumerate(zip(lst_img, self.bs)):
            if self.detector != 'rayonix':
                if self.detector == 'Pilatus900kw':
                    masks = self.det.calc_mask(bs=bs, bs_kind=self.bs_kind, optional_mask=optional_mask)
                    self.masks.append(masks[:, :195])
                    self.masks.append(masks[:, 212:212 + 195])
                    self.masks.append(masks[:, -195:])
                else:
                    self.masks.append(self.det.calc_mask(bs=bs, bs_kind=self.bs_kind, optional_mask=optional_mask))

            if self.detector == 'Pilatus1m':
                self.imgs.append(fabio.open(os.path.join(path, img)).data)
            elif self.detector == 'Pilatus900kw':
                # self.imgs.append(np.rot90(fabio.open(os.path.join(path, img)).data, 1))
                self.imgs.append(np.rot90(fabio.open(os.path.join(path, img)).data, 1)[:, :195])
                self.imgs.append(np.rot90(fabio.open(os.path.join(path, img)).data, 1)[:, 212:212 + 195])
                self.imgs.append(np.rot90(fabio.open(os.path.join(path, img)).data, 1)[:, -195:])

            elif self.detector == 'Pilatus300kw':
                self.imgs.append(np.rot90(fabio.open(os.path.join(path, img)).data, 1))
            elif self.detector == 'rayonix':
                self.imgs.append(np.rot90(fabio.open(os.path.join(path, img)).data, 1))
                self.masks.append(self.det.calc_mask(bs=bs, bs_kind=self.bs_kind, img=self.imgs[0]))
            elif self.detector == 'Pilatus100k_OPLS':
                self.imgs.append(fabio.open(os.path.join(path, img)).data)
            elif self.detector == 'Pilatus300k_OPLS':
                self.imgs.append(fabio.open(os.path.join(path, img)).data)
            elif self.detector == 'Pilatus800k_CMS':
                self.imgs.append(fabio.open(os.path.join(path, img)).data)

    def open_data_db(self, lst_img, optional_mask=None):
        """
        Function to load data directly a list of 2D array
        :param lst_img: list of 2D array containing the data. The data loaded together will be treated together as
        stitched images
        :param optional_mask: string. Can be 'tender' to mask extra chips of the detectors
        :return:
        """
        if self.detector is None:
            self.define_detector()
        if not lst_img:
            raise Exception('You are trying to load an empty dataset')
        if len(lst_img) != len(self.bs):
            self.bs = self.bs + [[0, 0]]*(len(lst_img) - len(self.bs))

        self.imgs = []
        for img, bs in zip(lst_img, self.bs):
            if self.detector != 'rayonix':
                self.masks.append(self.det.calc_mask(bs=bs, bs_kind=self.bs_kind, optional_mask=optional_mask))

            if self.detector == 'Pilatus1m':
                self.imgs.append(img)
            elif self.detector == 'Pilatus900kw':
                self.imgs.append(np.rot90(img, 1))
            elif self.detector == 'Pilatus300kw':
                self.imgs.append(np.rot90(img, 1))
            elif self.detector == 'rayonix':
                self.imgs.append(np.rot90(img, 1))
                self.masks.append(self.det.calc_mask(bs=bs, bs_kind=self.bs_kind, img=self.imgs[0]))
            elif self.detector == 'Pilatus100k_OPLS':
                self.imgs.append(img)
            elif self.detector == 'Pilatus300k_OPLS':
                self.imgs.append(img)
            elif self.detector == 'Pilatus800k_CMS':
                self.imgs.append(img)

    def calculate_integrator_trans(self, det_rots):
        self.ai = []
        ai = azimuthalIntegrator.AzimuthalIntegrator(**{'detector': self.det,
                                                        'rot1': 0,
                                                        'rot2': 0,
                                                        'rot3': 0}
                                                     )

        ai.setFit2D(self.sdd, self.center[0], self.center[1])
        ai.set_wavelength(self.wav)

        for i, det_rot in enumerate(det_rots):
            ai_temp = copy.deepcopy(ai)
            ai_temp.set_rot1(det_rot)
            self.ai.append(ai_temp)

    def calculate_integrator_gi(self, det_rots):
        ai = Transform(wavelength=self.wav, detector=self.det, incident_angle=self.alphai)
        ai.setFit2D(directDist=self.sdd, centerX=self.center[0], centerY=self.center[1])
        ai.set_incident_angle(self.alphai)

        for i, det_rot in enumerate(det_rots):
            ai_temp = copy.deepcopy(ai)
            ai_temp.set_rot1(det_rot)
            ai_temp.set_incident_angle(self.alphai)
            self.ai.append(ai_temp)

    def calculate_integrator_gi2(self, det_rots):
        self.ai = []
        ai = azimuthalIntegrator.AzimuthalIntegrator(**{'detector': self.det,
                                                        'rot1': 0,
                                                        'rot2': 0,
                                                        'rot3': 0}
                                                     )

        ai.setFit2D(self.sdd, self.center[0], self.center[1])
        ai.set_wavelength(self.wav)

        for i, det_rot in enumerate(det_rots):
            ai_temp = copy.deepcopy(ai)
            ai_temp.set_rot1(det_rot)
            self.ai.append(ai_temp)

    def stitching_data(self, flag_scale=True, interp_factor=1, perpendicular: bool = False, timing: bool = False):
        """
        Calculates the stitched image from the list of images and respective detector angles
        
        Parameters
        ----------
        flag_scale : bool, optional
            Boolean to scale or not consecutive images at different detector positions, by default True
        interp_factor : int, optional
            Factor to increase the binning of the stitching image. Can help fixing some mask issues, by default 1
        perpendicular : bool, optional
            If the sample has been mounted perpendicular to axis of detector rotation, corrects GI geometry, by default False.        
        timing : bool, optional
            If True, prints the time taken to calculate the integrator and stitch the images, by default False
        """
        
        self.img_st, self.qp, self.qz = [], [], []

        init = datetime.datetime.now()
            
        if self.ai == []:
            if len(self._cal_angles) != len(self.imgs):
                if self.detector != 'Pilatus900kw':
                    if len(self._cal_angles) !=0 and len(self._cal_angles) > len(self.imgs):
                        raise Exception('The number of angle for the %s is not good. '
                                        'There is %s images but %s angles' % (self.detector,
                                                                              int(len(self.imgs)),
                                                                              len(self._cal_angles)))

                    self.det_angles = [self.det_ini_angle + i * self.det_angle_step
                                       for i in range(0, len(self.imgs), 1)]

                else:
                    if len(self.det_angles) == 0:
                        # Setting the det angles also propogates to calibrated angles.
                        self.det_angles = [self.det_ini_angle + i * self.det_angle_step
                                           for i in range(0, int(len(self.imgs)//3), 1)]

                    if 3 * len(self._cal_angles) != len(self.imgs):
                        raise Exception('The number of angle for the %s is not good. '
                                        'There is %s images but %s angles' % (self.detector,
                                                                              int(len(self.imgs)//3),
                                                                              len(self._cal_angles)))

            # Calculate the self.ai values.
            if self.geometry == 'Transmission':
                self.calculate_integrator_trans(self._cal_angles)
            elif self.geometry == 'Reflection':
                self.calculate_integrator_gi(self._cal_angles)
            elif self.geometry == 'Reflection_test':
                self.calculate_integrator_gi2(self._cal_angles)
            else:
                raise Exception('Unknown geometry: should be either Transmission or Reflection')
            
            fin = datetime.datetime.now()
            if timing:
                print('Time to calculate the integrator: %s' % (fin - init))
            
            if perpendicular and not self._perpendicular_correction:
                init = datetime.datetime.now()
                # Reorder the images to reflect the perpendicular geometry
                if self.imgs and self.masks:
                    # Reverse the list order
                    self.imgs.reverse()
                    self.masks.reverse()
                    # Flip the image and mask; stiching is now done in the vertical direction for images.
                    for i in range(len(self.imgs)):
                        self.imgs[i] = np.fliplr(np.rot90(self.imgs[i], 1))
                        self.masks[i] = np.fliplr(np.rot90(self.masks[i], 1))
                
                # Correct the calculated integrators for perpendicular geometry; swap rot1 and rot2.
                for ai_i in self.ai:
                    ai_i.rot1, ai_i.rot2 = ai_i.rot2, ai_i.rot1
                    # rot 1 should become 0.
                
                # Reorder the angles to match the reordered images
                self.ai.reverse()
                
                fin = datetime.datetime.now()
                if timing:
                    print('Time to calc perpendicular geometry: %s' % (fin - init))
                
                # Set the perpendicular correction flag to True
                self._perpendicular_correction = True      
                
        else:
            # Correct the image and masks for perpendicular geometry, without defining new integrators.
            if perpendicular and not self._perpendicular_correction:
                init = datetime.datetime.now()
                if self.imgs and self.masks:
                    self.imgs.reverse()
                    self.masks.reverse()
                    for i in range(len(self.imgs)):
                        self.imgs[i] = np.fliplr(np.rot90(self.imgs[i], 1))
                        self.masks[i] = np.fliplr(np.rot90(self.masks[i], 1))
                fin = datetime.datetime.now()
                if timing:
                    print('Time to calc perpendicular geometry: %s' % (fin - init))
                    
                # Set the perpendicular correction flag to True
                self._perpendicular_correction = True
            

        init = datetime.datetime.now()
        self.img_st, self.mask_st, self.qp, self.qz, self.scales = stitch.stitching(self.imgs,
                                                                                    self.ai,
                                                                                    self.masks,
                                                                                    self.geometry,
                                                                                    flag_scale=flag_scale,
                                                                                    interp_factor=interp_factor
                                                                                    )
        fin = datetime.datetime.now()
        if timing:
            print('Time to stitch the images: %s' % (fin - init))

        if len(self.scales) == 1 or not flag_scale:
            pass
        elif len(self.scales) > 1:
            for i, scale in enumerate(self.scales):
                self.imgs[i] = self.imgs[i] / scale
        else:
            raise Exception('scaling waxs images error')
        

    def inpainting(self, **kwargs):
        self.inpaints, self.mask_inpaints = integrate1D.inpaint_saxs(self.imgs,
                                                                     self.ai,
                                                                     self.masks,
                                                                     **kwargs
                                                                     )

    def caking(self, radial_range=None, azimuth_range=None, npt_rad=500, npt_azim=500):
        if np.array_equal(self.img_st, []):
            self.stitching_data()

        if radial_range is None and 'Pilatus' in self.detector:
            radial_range = (0.01, np.sqrt(self.qp[1] ** 2 + self.qz[1] ** 2))
        if azimuth_range is None and 'Pilatus' in self.detector:
            azimuth_range = (-180, 180)

        if self.geometry == 'Transmission':
            if np.array_equal(self.inpaints, []):
                self.inpainting()
            self.cake, self.q_cake, self.chi_cake = integrate1D.cake_saxs(self.inpaints,
                                                                          self.ai,
                                                                          self.mask_inpaints,
                                                                          radial_range=radial_range,
                                                                          azimuth_range=azimuth_range,
                                                                          npt_rad=npt_rad,
                                                                          npt_azim=npt_azim
                                                                          )
        elif self.geometry == 'Reflection':
            #ToDo implement a way to modify the dimension of the cake if required (it need to match the image dim ratio)
            # if self.inpaints == []:
            #     self.inpainting()
            self.cake, self.q_cake, self.chi_cake = integrate1D.cake_gisaxs(self.img_st,
                                                                            self.qp,
                                                                            self.qz,
                                                                            bins=None,
                                                                            radial_range=radial_range,
                                                                            azimuth_range=azimuth_range
                                                                            )

    def radial_averaging(self, radial_range=None, azimuth_range=None, npt=2000):
        self.q_rad, self.I_rad = [], []
        # Also do error propogation
        self.q_rad_err, self.I_rad_err = [], []

        if self.geometry == 'Transmission':
            if np.array_equal(self.inpaints, []):
                self.inpainting()
            if radial_range is None and (self.detector == 'Pilatus300kw' or self.detector == 'Pilatus900kw'):
                radial_range = (0.001, np.sqrt(self.qp[1]**2 + self.qz[1]**2))
            if azimuth_range is None and (self.detector == 'Pilatus300kw' or self.detector == 'Pilatus900kw'):
                azimuth_range = (0, 90)

            if radial_range is None and self.detector == 'Pilatus1m':
                radial_range = (0.0001, np.sqrt(self.qp[1]**2 + self.qz[1]**2))
            if azimuth_range is None and self.detector == 'Pilatus1m':
                azimuth_range = (-180, 180)

            self.q_rad, self.I_rad, self.I_rad_err = integrate1D.integrate_rad_saxs(self.inpaints,
                                                                    self.ai,
                                                                    self.masks,
                                                                    radial_range=radial_range,
                                                                    azimuth_range=azimuth_range,
                                                                    npt=npt
                                                                    )

        elif self.geometry == 'Reflection':
            if np.array_equal(self.img_st, []):
                self.stitching_data()
            if radial_range is None and 'Pilatus' in self.detector:
                radial_range = (0, self.qp[1])
            if azimuth_range is None and 'Pilatus' in self.detector:
                azimuth_range = (0, self.qz[1])

            if radial_range is None and self.detector == 'rayonix':
                radial_range = (0, self.qp[1])
            if azimuth_range is None and self.detector == 'rayonix':
                azimuth_range = (0, self.qz[1])

            self.q_rad, self.I_rad, self.I_rad_err = integrate1D.integrate_rad_gisaxs(self.img_st,
                                                                      self.qp,
                                                                      self.qz,
                                                                      bins=npt,
                                                                      radial_range=radial_range,
                                                                      azimuth_range=azimuth_range)

        else:
            raise Exception('Unknown geometry: should be either Transmission or Reflection')

    def azimuthal_averaging(self, radial_range=None, azimuth_range=None, npt_rad=500, npt_azim=500):
        self.chi_azi, self.I_azi = [], []
        if radial_range is None and (self.detector == 'Pilatus300kw' or self.detector == 'Pilatus900kw'):
            radial_range = (0.01, np.sqrt(self.qp[1] ** 2 + self.qz[1] ** 2))
        if azimuth_range is None and (self.detector == 'Pilatus300kw' or self.detector == 'Pilatus900kw'):
            azimuth_range = (1, 90)

        if radial_range is None and self.detector == 'Pilatus1m':
            radial_range = (0.001, np.sqrt(self.qp[1] ** 2 + self.qz[1] ** 2))
        if azimuth_range is None and self.detector == 'Pilatus1m':
            azimuth_range = (-180, 180)

        if np.array_equal(self.cake, []):
            self.caking(radial_range=radial_range,
                        azimuth_range=azimuth_range,
                        npt_rad=npt_rad,
                        npt_azim=npt_azim
                        )

        self.chi_azi, self.I_azi = integrate1D.integrate_azi_saxs(self.cake,
                                                                  self.q_cake,
                                                                  self.chi_cake,
                                                                  radial_range=radial_range,
                                                                  azimuth_range=azimuth_range
                                                                  )

    def horizontal_integration(self, q_per_range=None, q_par_range=None):
        if np.array_equal(self.img_st, []):
            self.stitching_data()

        self.q_hor, self.I_hor = integrate1D.integrate_qpar(self.img_st,
                                                            self.qp,
                                                            self.qz,
                                                            q_par_range=q_par_range,
                                                            q_per_range=q_per_range
                                                            )

    def vertical_integration(self, q_per_range=None, q_par_range=None):
        if np.array_equal(self.img_st, []):
            self.stitching_data()

        self.q_ver, self.I_ver = integrate1D.integrate_qper(self.img_st,
                                                            self.qp,
                                                            self.qz,
                                                            q_par_range=q_par_range,
                                                            q_per_range=q_per_range
                                                            )
