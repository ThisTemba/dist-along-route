import numpy as np
from scipy.spatial import KDTree


def subdivide_path(path_pts: np.ndarray, max_dist: float) -> np.ndarray:
    """
    Subdivide a 2D path so no two points are more than max_dist apart.
    """
    subdivided_pts = []
    for i in range(len(path_pts) - 1):
        # for each segment, get the start and end points of the segment
        p1, p2 = path_pts[i : i + 2]
        dist = np.linalg.norm(p2 - p1)

        if dist <= max_dist:
            # if no need to subdivide, just add the start point
            subdivided_pts.append(p1)
        else:
            # if need to subdivide, add the points along the segment
            num_divisions = int(np.ceil(dist / max_dist))
            segment_fractions = np.linspace(0, 1, num_divisions + 1)[:-1]
            subdivided_pts.extend(p1 + segment_fractions[:, np.newaxis] * (p2 - p1))

    subdivided_pts.append(path_pts[-1])

    return np.array(subdivided_pts)


class Path:
    """
    This class quickly computes the distance of many test points from the start of a path.
    The test points are near the path, not on the path.
    It does this by first subdividing the path into segments with size = "precision".
    Then it creates a KDTree of the subdivided path.
    We then compute the distance from the start of the path to each point on the path.

    To compute the distance from any test point to the true endpoint, we:
        1. find the closest point on the path to the test point
        2. look up the distance from that point to the path start

    Cons:
        1. distance estimate can only be as good as the specified precision
        2. initializes slowly for long, highly subdivided paths

    Pros:
        1. Extremely fast distance estimates for large number of test points
        2. Rather straightforward to understand what is being done

    Notes:
        1. does not take into account perpendicular distance to the path of test points
        2. path is not smoothed during subdivision
        3. optimized for bulk processing of data

    """

    def __init__(self, path_pts_original: np.ndarray, precision: float):
        assert type(path_pts_original) is np.ndarray
        assert path_pts_original.shape[0] >= 2  # at least two points
        assert path_pts_original.shape[1] == 2  # 2D points
        assert type(precision) is float or type(precision) is int
        assert precision > 0

        self.path_pts_original = path_pts_original
        self.path_pts = subdivide_path(path_pts_original, precision)
        self.tree = KDTree(self.path_pts)
        self.dists_from_start = self._get_path_dists_from_start()

    def _get_path_dists_from_start(self) -> np.ndarray:
        # distance between consecutive points
        pt_to_pt_dists = np.linalg.norm(np.diff(self.path_pts, axis=0), axis=1)
        # cumulative sum of distances between consecutive points (distances of each point from the start)
        # insert 0 at the beginning to account for the first point
        dists_from_start = np.insert(np.cumsum(pt_to_pt_dists), 0, 0)
        return dists_from_start

    def get_dist_from_start(self, pt: np.ndarray) -> np.ndarray:
        assert type(pt) is np.ndarray

        # Handle both single coord and array of coords
        if len(pt.shape) == 1:
            assert pt.shape[0] == 2
            pt = pt.reshape(1, 2)
        else:
            assert pt.shape[1] == 2

        # find the closest point on the path
        closest_idx = self.tree.query(pt)[1]

        dists_from_start = self.dists_from_start[closest_idx]

        dists_from_start[closest_idx == 0] = np.nan
        dists_from_start[closest_idx == len(self.path_pts) - 1] = np.nan

        # Return scalar for single input
        if len(pt) == 1:
            return dists_from_start[0]

        return dists_from_start
