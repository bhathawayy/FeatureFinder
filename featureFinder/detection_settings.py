class DefaultDetection:
    """
    Default detection settings. Note, the area or size parameters can typically be estimated using ImageJ. Fit a
    shape to your feature then "Measure" to return the area in pxl^2.

    USE THE HELPER GUI TO CONFIGURE!
    """
    # Parameters used for detection algorithms
    circle_size: tuple = (1000, 6000)  # Expected size of fiducial [(pxl^2, pxl^2)]
    circularity_min: float = 0.8  # The closer to 1, the more "perfect" the circle is
    default_pivot_point: tuple | None = (1800, 1050)  # If None, image center [(pxl, pxl)]
    deviation_cutoff: int = 100  # Cutoff distance between multiple detections [pxl]
    gauss: int = 11  # Gaussian blur kernel size
    hough_min_length: int = 30  # Expected length of detected (CRH) line [pxl]
    rect_size: tuple = (10000, 30000)  # Expected size of feature (non-fiducial) [(pxl^2, pxl^2)]
    threshold: int = 30  # Explicit edge thresholding

    # Reference filters (used for removing detections based on expected field point locations)
    ang2pxl: int = 65  # Angle to pixel conversion for reference points
    default_clock_angle: float | int = 0.0  # Angle of camera under test
    fid_triangle_area: float | int = 0  # Expected size of triangle formed by fiducials [pxl^2]
    orientation_left: list = [False, False, 0.0]  # Left eye reference: [X Flip, Y Flip, Rotation [deg]]
    orientation_right: list = [False, False, 0.0]  # Right eye reference: [X Flip, Y Flip, Rotation [deg]]

    def __init__(self, **kwargs):
        """
        Initiate the class on execution.

        :param kwargs: test_system (str) and/or is_crosshair (bool)
        """
        self.gauss: int = self._even_to_odd(self.gauss)
        self._limit_fiducial_size()

    def set_test_system_specifics(self, test_system: str):
        """
        Optional abstract method for setting test system specific settings.

        :param test_system: Acronym for test system.
        """
        pass

    @staticmethod
    def _even_to_odd(val: float | int) -> int:
        """
        Ensure variable is greater than 0 and odd.

        :param val: Value to check.
        :return: Odd integer, greater than 0.
        """
        if val < 0:
            val = 1
        elif int(val) % 2 == 0:
            val += 1

        return int(val)

    def _limit_fiducial_size(self):
        """
        Ensure fiducial size is smaller than the defined feature size.
        """
        new_min, new_max = self.circle_size
        if self.circle_size[1] > self.rect_size[0]:
            new_max = self.rect_size[0]
            if self.circle_size[0] > new_max:
                new_min = 0
            self.circle_size: tuple = (new_min, new_max)

    def _check_kwargs(self, **kwargs):
        """
        Check if there are optional arguments passed and do actions accordingly.

        :param kwargs: Optional argument: test_system
        """
        if "test_system" in kwargs:
            self.set_test_system_specifics(kwargs["test_system"])


# Configured child classes (DO NOT REMOVE/EDIT) --------------------------------------------------------------------- #
class GalileoDetection(DefaultDetection):  # Tested: CRH, G20 SFR, JOHNNY5, uBAT, TODO: G30 SFR

    def __init__(self, **kwargs):
        """
        Detection settings for Galileo 30deg and 20deg.
        """
        super().__init__()

        self.ang2pxl: int = 65
        self.circle_size: tuple = (1000, 5000)
        self.circularity_min: float = 0.7
        self.default_clock_angle: float | int = -5
        self.default_pivot_point: tuple | None = (1900, 1100)
        self.deviation_cutoff: int = 150
        self.fid_triangle_area: float | int = 9.8
        self.gauss: int = 1
        self.hough_min_length: int = 50
        self.orientation_left: list = [False, False, 0.0]
        self.orientation_right: list = [False, False, 0.0]
        self.rect_size: tuple = (10000, 50000)
        self.threshold: int = 10

        self._check_kwargs(**kwargs)

    def set_test_system_specifics(self, test_system: str):
        """
        Optional abstract method for setting test system specific settings.

        :param test_system: Acronym for test system.
        """
        if test_system == "JOHNNY5":
            self.hough_min_length: int = 40
        elif test_system == "BAT":
            self.default_pivot_point: tuple | None = None
            self.orientation_right: list = [True, False, 0.0]


class MidasDetection(DefaultDetection):  # Tested: PIN

    def __init__(self, **kwargs):
        """
        Detection settings for Midas.
        """
        super().__init__()

        self.ang2pxl: int = 65
        self.circle_size: tuple = (1000, 6000)
        self.circularity_min: float = 0.8
        self.default_clock_angle: float | int = -5
        self.default_pivot_point: tuple | None = None
        self.deviation_cutoff: int = 200
        self.fid_triangle_area: float | int = 11.8
        self.gauss: int = 89
        self.hough_min_length: int = 0
        self.orientation_left: list = [True, False, -90.0]
        self.orientation_right: list = [False, False, 90.0]
        self.rect_size: tuple = (15000, 35000)
        self.threshold: int = 100

        self._check_kwargs(**kwargs)

    def set_test_system_specifics(self, test_system: str):
        """
        Optional abstract method for setting test system specific settings.

        :param test_system: Acronym for test system.
        """
        if test_system == "PIN":
            self.ang2pxl: int = 80
            self.circle_size: tuple = (1000, 10000)
            self.circularity_min: float = 0.3
            self.default_clock_angle: float | int = 0
            self.fid_triangle_area: float | int = 12.4
            self.gauss: int = 201
            self.orientation_left: list = [True, False, 0.0]
            self.orientation_right: list = [True, True, 0.0]


class ML2Detection(DefaultDetection):  # Tested: BIG, TET

    def __init__(self, **kwargs):
        """
        Detection settings for ML2.
        """
        super().__init__()

        self.ang2pxl: int = 80
        self.circle_size: tuple = (1000, 10000)
        self.circularity_min: float = 0.2
        self.default_clock_angle: float | int = -5
        self.default_pivot_point: tuple | None = None
        self.deviation_cutoff: int = 150
        self.fid_triangle_area: float | int = 58.3
        self.gauss: int = 1
        self.orientation_left: list = [False, True, 90]
        self.orientation_right: list = [True, True, 90]
        self.rect_size: tuple = (10000, 50000)
        self.threshold: int = 30

        self._check_kwargs(**kwargs)

    def set_test_system_specifics(self, test_system: str):
        """
        Optional abstract method for setting test system specific settings.

        :param test_system: Acronym for test system.
        """
        if test_system == "TET":
            self.ang2pxl: int = 65
            self.fid_triangle_area: float | int = 59.3
            self.orientation_left: list = [False, False, 90]
            self.orientation_right: list = [False, False, 90]


class HydraDetection(DefaultDetection):  # Tested

    def __init__(self, **kwargs):
        """
        Detection settings for Hydra.
        """
        super().__init__()

        self.ang2pxl: int = 78
        self.circle_size: tuple = (0, 0)
        self.circularity_min: float = 0
        self.default_clock_angle: float | int = 0
        self.default_pivot_point: tuple | None = None
        self.deviation_cutoff: int = 200
        self.fid_triangle_area: float | int = 0
        self.gauss: int = 21
        self.hough_min_length: int = 15
        self.orientation_left: list = [False, False, 0.0]
        self.orientation_right: list = [False, False, 0.0]
        self.rect_size: tuple = (10000, 30000)
        self.threshold: int = 30

        self._check_kwargs(**kwargs)


class TriOpticsDetection(DefaultDetection):  # Tested

    def __init__(self, **kwargs):
        """
        Detection settings for Olaf.
        """
        super().__init__()

        self.default_pivot_point: tuple | None = None
        self.deviation_cutoff: int = 250
        self.gauss: int = 1
        self.hough_min_length: int = 500
        self.orientation_left: list = [False, False, 0.0]
        self.orientation_right: list = [False, False, 0.0]
        self.threshold: int = 10

        self._check_kwargs(**kwargs)
