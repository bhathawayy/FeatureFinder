import json
import math
import time

import cv2
from scipy import spatial

from featureFinder.__init__ import *
from internal_objects import *


class ImageLoader:
    """
    A class for loading images into an internal dictionary, processing them, and collecting metadata.
    """

    def __init__(self, image_array_or_file: np.ndarray | str, is_crosshair: bool = False):
        """
        Load image into internal dictionary. This will load the image as an array and collect info from name if
        standard formatting.

        :param image_array_or_file: Path to image folder, image file, or image array.
        :param is_crosshair: Boolean flag used for detection and MTF calculations.
        """
        # Init class variable(s)
        self._image_array_or_file: str | np.ndarray = image_array_or_file
        self._image_dir: str | None = None  # TODO: needed?
        self._image_name: str | None = None
        self.is_crosshair: bool = is_crosshair
        self.processed_info: ProcessedInfo = ProcessedInfo()

    def load_image(self) -> ProcessedInfo:
        """
        Check the type and validity of the input image or directory.

        :return: ProcessedInfo object containing image info, converted arrays, and detection info.
        """
        # Check type of input
        if isinstance(self._image_array_or_file, np.ndarray) and self._image_array_or_file.size > 0:
            image_array = self._image_array_or_file
            image_dir = None
            image_name = None
        elif isinstance(self._image_array_or_file, str) and os.path.isfile(self._image_array_or_file):
            image_array = cv2.imread(self._image_array_or_file, cv2.IMREAD_UNCHANGED)
            image_dir = os.path.dirname(self._image_array_or_file)
            image_name = os.path.basename(self._image_array_or_file)
        else:
            raise FileNotFoundError("Invalid image array or path!")

        # Save info to internal object
        self.processed_info.info.file_name = image_name
        self.processed_info.info.directory = image_dir

        # Load the image into supported arrays
        start = time.time()
        self.processed_info.arrays = self._array_to_supported_arrays(image_array)
        print("Image Loading runtime (s):", round(time.time() - start, 3))

        return self.processed_info

    @staticmethod
    def _array_to_supported_arrays(image_array: np.ndarray) -> SupportedArrays:
        """
        Extract monochrome 8-bit, monochrome 16-bit, and color 8-bit arrays.

        :param image_array: Array to convert.
        :return: supported arrays as SupportedArrays object
        """
        supported: SupportedArrays = SupportedArrays()
        if image_array.size:
            supported.mono8 = convert_color_bit(image_array, color_channels=1, out_bit_depth=8)
            supported.mono16 = convert_color_bit(image_array, color_channels=1, out_bit_depth=16)
            supported.color8 = convert_color_bit(image_array, color_channels=3, out_bit_depth=8)

        return supported


def check_path(target_path: str, overwrite: bool = True) -> str:
    """
    Checks if directory and path exists, if not it creates one.
    :param overwrite: Overwrite the file (Ture) or not (False).
    :param target_path: Path to file.
    :return: Unique path.
    """
    # Get directory path
    target_split = os.path.splitext(target_path)
    if len(target_split[1]) > 0:
        target_dir = os.path.dirname(target_path)
        target_file = os.path.basename(target_path)
    else:
        target_dir = target_split[0]
        target_file = target_split[1]

    # Check validity of directory
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
        except PermissionError:
            target_dir = os.path.join(os.getcwd(), "Debug")
            print(f"Lacking write permissions. Saving locally instead: {target_dir}")
        except FileExistsError:
            pass
    target_path = os.path.join(target_dir, target_file)

    # Check if path already exists, add (#) to name
    if not overwrite and os.path.isfile(target_path):
        count = 1
        temp = target_file.split(".")  # split by .
        file_name, file_ext = (".".join(temp[:-1]), temp[-1])  # join back all but the last entry (the extension)
        new_file_name = "%s (%i).%s" % (file_name, count, file_ext)
        while os.path.exists(os.path.join(target_dir, new_file_name)):
            count += 1
            new_file_name = "%s (%i).%s" % (file_name, count, file_ext)
        target_path = os.path.join(target_dir, new_file_name)

    return target_path


def convert_color_bit(image: np.ndarray | str, color_channels: int = None, out_bit_depth: int = None,
                      in_bit_depth: int = None) -> np.ndarray:
    """
    Converts image array into RGB/Monochrome with specified bit-depth.
    :param color_channels: Color descriptor options: 3 = RGB, 1 = Monochrome
    :param image: Image array to be processed.
    :param in_bit_depth: Bit-depth options: 8, 12, 16
    :param out_bit_depth: Bit-depth options: 8, 16
    :return: Converted image array.
    """
    # Check input image
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    converted_array = np.array(image).copy()

    if converted_array.size != 0:
        # Initialize local variables
        if in_bit_depth is None:
            in_bit_depth = converted_array.dtype.name
        else:
            in_bit_depth = str(in_bit_depth)
        current_shape = converted_array.shape

        # Convert to desired bit-depth
        if out_bit_depth is not None:
            if "16" in in_bit_depth and "8" in str(out_bit_depth):  # 16-bit to 8-bit
                converted_array = converted_array.astype(float)
                converted_array = (converted_array / (2 ** 8)).astype('uint8')
            elif "12" in in_bit_depth and "16" in str(out_bit_depth):  # 12-bit to 16-bit
                converted_array = converted_array.astype(float)
                converted_array = (converted_array / (2 ** 4)).astype('uint8')
            elif "8" in in_bit_depth and "16" in str(out_bit_depth):  # 8-bit to 16-bit
                converted_array = converted_array.astype(float)
                converted_array = (converted_array * (2 ** 8)).astype('uint16')

        # Convert to desired color
        if color_channels is not None:
            if len(current_shape) == 2 and color_channels == 3:
                converted_array = cv2.cvtColor(converted_array, cv2.COLOR_GRAY2RGB)
            elif len(current_shape) == 3 and color_channels == 1:
                converted_array = cv2.cvtColor(converted_array, cv2.COLOR_RGB2GRAY)

    return converted_array


def get_nearest_point(check_point: tuple, references: list) -> tuple[int, float]:
    """
    Find nearest point in list of points to reference.
    :param check_point: Point to reference.
    :param references: List of points to look in.
    :return: Index of point in references closest to check_point.
    """
    tree = spatial.KDTree(references)
    distance, index = tree.query([check_point])

    # Find the minimum distance
    min_distance = float(distance[0])

    # Get all indices with the same minimum distance
    nearest_indices = [i for i, ref in enumerate(references) if
                       spatial.distance.euclidean(check_point, ref) == min_distance]
    min_index = min(nearest_indices)

    return min_index, min_distance


def get_point_distance(point1: tuple | list | np.ndarray, point2: tuple | list | np.ndarray) -> float:
    """
    Get pixel distance between two coordinate points.
    :param point1: First point coordinates.
    :param point2: Second point coordinates.
    :return: Distance between points and whether this distance is less than cutoff.
    """
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    return distance


def crop_image(image_array_mono: np.ndarray, roi_center: tuple, roi_size_wh: tuple = (100, 100)) -> np.ndarray:
    """
    Crop input image with square ROI.
    :param image_array_mono: Image array to be processed.
    :param roi_center: ROI center point.
    :param roi_size_wh: ROI box width and height in pixels.
    :return: Cropped image.
    """
    cropped_image = np.array([])
    roi_width, roi_height = roi_size_wh
    if image_array_mono.size:
        left = int(roi_center[0] - roi_width / 2)
        top = int(roi_center[1] - roi_height / 2)
        right = int(roi_center[0] + roi_width / 2)
        bottom = int(roi_center[1] + roi_height / 2)
        cropped_image = image_array_mono.copy()[top:bottom, left:right]

    return cropped_image


def get_pxl_midpoint(point1: tuple | list | np.ndarray, point2: tuple | list | np.ndarray) -> tuple[int, int]:
    """
    Get the midpoint between two points.
    :param point1: Point 1.
    :param point2: Point 2.
    :return: Midpoint.
    """
    x_point = int((point1[0] + point2[0]) / 2)
    y_point = int((point1[1] + point2[1]) / 2)

    return x_point, y_point


def get_point_angle(point1: tuple | list | np.ndarray, point2: tuple | list | np.ndarray) -> float:
    """
    Reports angle between two cartesian points.
    :param point1: Point 1.
    :param point2: Point 2.
    :return: Angle in degrees.
    """
    angle_radians = math.atan2(point2[1] - point1[1], point2[0] - point1[0])
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


def read_json_file(file_path: str) -> dict[str, any]:
    """
    Reads the JSON file at the specified path and returns the contained data.
    :param file_path: Path to JSON file
    :return: Data in JSON file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Invalid path: {file_path}")
    with open(file_path) as json_file:
        data = json.load(json_file)

    return data


def save_image(image_path: str, image_array: np.ndarray, overwrite: bool = False) -> None:
    """
    Saves drawn image for debugging.
    :param image_path: Where to save the image.
    :param image_array: Image to be saved.
    :param overwrite: Overwrite an existing file with the same name (True) or not (False)?
    :return: None
    """
    if image_array is not None and image_array.size > 0:
        # Check the image path
        checked_path = check_path(image_path, overwrite=overwrite)

        # Save the image
        try:
            if not os.path.isdir(os.path.dirname(checked_path)):
                os.makedirs(os.path.dirname(checked_path))
            cv2.imwrite(checked_path, image_array)
        except PermissionError:
            local_path = os.path.join(os.getcwd(), os.path.basename(image_path))
            print(f"Lacking write permissions for this directory. Saving locally instead: {local_path}")
            cv2.imwrite(local_path, image_array)
