from typing import Literal, Tuple, Union

import numpy as np
from coordstometers import coords_to_meters

from .path import _Path

DistanceUnit = Literal["meters", "feet", "miles", "kilometers"]

# Conversion factors from meters to other units
CONVERSION_FACTORS = {
    "meters": 1.0,
    "feet": 3.28084,
    "miles": 1 / 1609.344,
    "kilometers": 1 / 1000,
}


def convert_from_meters(pts_meters: np.ndarray, unit: DistanceUnit) -> np.ndarray:
    """
    Convert distances from meters to the specified unit.

    Args:
        pts_meters: Array of distances in meters
        unit: Target unit for conversion

    Returns:
        Array of distances in the specified unit

    Raises:
        ValueError: If unit is not supported
    """
    if unit not in CONVERSION_FACTORS:
        raise ValueError(
            f"Unsupported unit: {unit}. Supported units: {list(CONVERSION_FACTORS.keys())}"
        )

    return pts_meters * CONVERSION_FACTORS[unit]


class Route:
    """
    A class to compute distances along a route from latitude/longitude coordinates.

    This class takes a series of lat/lon points representing a route and provides
    methods to compute distances from the start of the route for any given point.
    """

    def __init__(
        self,
        lat_lon_pts: np.ndarray,
        precision: float = 1.0,
        distance_unit: DistanceUnit = "feet",
    ) -> None:
        """
        Initialize a Route object.

        Args:
            lat_lon_pts: Array of shape (n, 2) containing lat/lon coordinates
            precision: Maximum distance between subdivided path points
            distance_unit: Unit for distance calculations

        Raises:
            ValueError: If lat_lon_pts is invalid or precision is non-positive
        """
        self._validate_inputs(lat_lon_pts, precision)

        self.lat_lon_pts = lat_lon_pts
        self.precision = precision
        self.distance_unit: DistanceUnit = distance_unit
        self.center_coord = self._get_center_coord()
        self.path = self._get_route()

    @staticmethod
    def _validate_inputs(lat_lon_pts: np.ndarray, precision: float) -> None:
        """Validate input parameters."""
        if not isinstance(lat_lon_pts, np.ndarray) or lat_lon_pts.shape[1] != 2:
            raise ValueError("lat_lon_pts must be a numpy array with shape (n, 2)")

        if lat_lon_pts.shape[0] < 2:
            raise ValueError("At least 2 points are required to define a route")

        if precision <= 0:
            raise ValueError("Precision must be positive")

    def _get_center_coord(self) -> np.ndarray:
        """
        Calculate the center coordinate of the route bounding box.

        Returns:
            Center coordinate as [lat, lon]
        """
        mins = np.min(self.lat_lon_pts, axis=0)
        maxs = np.max(self.lat_lon_pts, axis=0)
        return (mins + maxs) / 2

    def _get_route(self) -> _Path:
        """
        Convert lat/lon coordinates to meters and create a Path object.

        Returns:
            Path object for distance calculations
        """
        pts_meters = coords_to_meters(self.lat_lon_pts, self.center_coord)
        pts_new_unit = convert_from_meters(pts_meters, self.distance_unit)
        return _Path(pts_new_unit, self.precision)

    def get_dist_from_start(
        self, lat_lon_pt: np.ndarray, max_dist_from_route: Union[float, None] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the distance from the start of the route for given lat/lon point(s).
        Output is in the same unit as the route.

        Args:
            lat_lon_pt: Single point as [lat, lon] or array of points with shape (n, 2)
            max_dist_from_route: Maximum perpendicular distance from route.
                               Points beyond this distance will return NaN.

        Returns:
            Tuple of (distances_from_start, distances_to_route) in the same unit as the route
        """
        pts_meters = coords_to_meters(lat_lon_pt, self.center_coord)
        pts = convert_from_meters(pts_meters, self.distance_unit)
        return self.path.get_dist_from_start(pts, max_dist_from_route)
