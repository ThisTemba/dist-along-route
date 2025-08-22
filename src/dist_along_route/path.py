from typing import Tuple, Union

import numpy as np
from scipy.spatial import KDTree


def subdivide_path(path_pts: np.ndarray, max_dist: float) -> np.ndarray:
    """
    Subdivide a 2D path so no two points are more than max_dist apart.

    Args:
        path_pts: Array of shape (n, 2) containing 2D path points
        max_dist: Maximum distance between consecutive points

    Returns:
        Array of subdivided path points

    Raises:
        ValueError: If max_dist is not positive
    """
    if max_dist <= 0:
        raise ValueError("max_dist must be positive")

    subdivided_pts = []
    for i in range(len(path_pts) - 1):
        p1, p2 = path_pts[i : i + 2]
        dist = np.linalg.norm(p2 - p1)

        if dist <= max_dist:
            subdivided_pts.append(p1)
        else:
            num_divisions = int(np.ceil(dist / max_dist))
            t_values = np.linspace(0, 1, num_divisions + 1)[:-1]
            subdivided_pts.extend(p1 + t_values[:, np.newaxis] * (p2 - p1))

    subdivided_pts.append(path_pts[-1])
    return np.array(subdivided_pts)


class _Path:
    """
    A class to quickly compute distances of test points from the start of a path.

    This class efficiently computes distances for many test points near (but not on)
    a path by:
    1. Subdividing the path into segments with size = "precision"
    2. Creating a KDTree of the subdivided path
    3. Computing distances from the start to each point on the path
    4. For any test point, finding the closest point on the path and looking up
       its distance from the start

    Pros:
        - Extremely fast distance estimates for large numbers of test points
        - Straightforward implementation and understanding

    Cons:
        - Distance estimate accuracy limited by specified precision
        - Slow initialization for long, highly subdivided paths
        - Does not account for perpendicular distance to path
        - Path is not smoothed during subdivision

    Notes:
        - Optimized for bulk processing of data
        - Returns NaN for points at path endpoints or beyond max_dist_from_route
    """

    def __init__(self, path_pts_original: np.ndarray, precision: float) -> None:
        """
        Initialize a Path object.

        Args:
            path_pts_original: Array of shape (n, 2) containing 2D path points
            precision: Maximum distance between subdivided path points

        Raises:
            ValueError: If inputs are invalid
        """
        self._validate_inputs(path_pts_original, precision)

        self.precision = precision
        self.path_pts_original = path_pts_original
        self.path_pts = subdivide_path(path_pts_original, self.precision)
        self.tree = KDTree(self.path_pts)
        self.dists_from_start = self._get_path_dists_from_start()

    @staticmethod
    def _validate_inputs(path_pts_original: np.ndarray, precision: float) -> None:
        """Validate input parameters."""
        if not isinstance(path_pts_original, np.ndarray):
            raise ValueError("path_pts_original must be a numpy array")

        if (
            path_pts_original.shape != (path_pts_original.shape[0], 2)
            or path_pts_original.shape[0] < 2
        ):
            raise ValueError("path_pts_original must have shape (n, 2) with n >= 2")

        if not isinstance(precision, (float, int)) or precision <= 0:
            raise ValueError("precision must be a positive number")

    def _get_path_dists_from_start(self) -> np.ndarray:
        """
        Calculate cumulative distances from the start of the path to each point.

        Returns:
            Array of cumulative distances from the start
        """
        segment_dists = np.linalg.norm(np.diff(self.path_pts, axis=0), axis=1)
        return np.insert(np.cumsum(segment_dists), 0, 0)

    def _warn_for_distant_points(self, dists_to_route: np.ndarray) -> None:
        """
        Print a warning if some points are very far from the route.

        Args:
            dists_to_route: Array of distances from test points to the route
        """
        total_route_length = self.dists_from_start[-1]
        if np.any(dists_to_route > total_route_length):
            print("Warning: Some points are very far from the route")
            print(f"Total route length: {total_route_length} units")
            print(f"Furthest point is {np.max(dists_to_route)} units from the route")

    def get_dist_from_start(
        self, pt: np.ndarray, max_dist_from_route: Union[float, None] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the distance from the start of the path for given test point(s).

        Args:
            pt: Single point as [x, y] or array of points with shape (n, 2)
            max_dist_from_route: Maximum perpendicular distance from route.
                               Points beyond this distance will return NaN.

        Returns:
            Tuple of (distances_from_start, distances_to_route)

        Raises:
            ValueError: If pt is not a valid numpy array
        """
        if not isinstance(pt, np.ndarray):
            raise ValueError("pt must be a numpy array")

        # Handle both single coordinate and array of coordinates
        if len(pt.shape) == 1:
            if pt.shape[0] != 2:
                raise ValueError("Single point must have shape (2,)")
            pt = pt.reshape(1, 2)
        else:
            if pt.shape[1] != 2:
                raise ValueError("Point array must have shape (n, 2)")

        # Find the closest point on the path
        dists_to_route, indices = self.tree.query(pt)
        dists_from_start = self.dists_from_start[indices]

        # Create mask for points to exclude (endpoints or too far)
        nan_mask = (indices == 0) | (indices == len(self.path_pts) - 1)
        if max_dist_from_route is not None:
            nan_mask |= dists_to_route > max_dist_from_route

        # Apply mask
        dists_from_start[nan_mask] = np.nan
        dists_to_route[nan_mask] = np.nan

        self._warn_for_distant_points(dists_to_route)

        # Return scalar for single input
        if len(pt) == 1:
            return dists_from_start[0], dists_to_route[0]

        return dists_from_start, dists_to_route
