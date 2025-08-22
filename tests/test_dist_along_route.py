import time
from typing import Any, cast

import numpy as np
import pytest

from dist_along_route import Path, Route, subdivide_path


def test_subdivide_path_no_subdivision_when_within_max_dist():
    path_pts = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    max_dist = 0.6
    out = subdivide_path(path_pts, max_dist)
    assert out.shape == (3, 2)
    assert np.allclose(out, path_pts)


def test_subdivide_path_single_segment_exact_division():
    path_pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    max_dist = 0.25  # 1.0 / 0.25 = 4 segments -> 3 interior points
    out = subdivide_path(path_pts, max_dist)
    # expected points at 0.0, 0.25, 0.5, 0.75, 1.0
    expected = np.array([[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0]])
    assert np.allclose(out, expected)


def test_subdivide_path_single_segment_non_exact_division():
    path_pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    max_dist = 0.3  # ceil(1/0.3)=4 -> 3 interior points at 0.25, 0.5, 0.75
    out = subdivide_path(path_pts, max_dist)
    expected = np.array([[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0]])
    assert np.allclose(out, expected)


def test_subdivide_path_multi_segment_mixed():
    path_pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    max_dist = 0.6
    out = subdivide_path(path_pts, max_dist)
    # First segment length 1.0 -> ceil(1/0.6)=2 -> insert 1 point at 0.5
    # Second segment length 2.0 -> ceil(2/0.6)=4 -> insert 3 points at y=0.5,1.0,1.5
    expected = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, 1.5],
            [1.0, 2.0],
        ]
    )
    assert np.allclose(out, expected)


def test_path_init_validation_errors():
    with pytest.raises(AssertionError):
        Path(cast(Any, [[0.0, 0.0], [1.0, 0.0]]), 0.1)  # not np.ndarray
    with pytest.raises(AssertionError):
        Path(np.array([[0.0, 0.0]]), 0.1)  # less than 2 points
    with pytest.raises(AssertionError):
        Path(np.array([[0.0], [1.0]]), 0.1)  # not 2D
    with pytest.raises(AssertionError):
        Path(np.array([[0.0, 0.0], [1.0, 0.0]]), 0)  # precision not > 0


def test_get_path_dists_from_start_basic():
    # Simple right angle path: (0,0)->(1,0)->(1,2)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    path = Path(pts, 10.0)  # no subdivision due to large precision
    # Distances from start for original points: 0,1,3
    assert np.allclose(path.dists_from_start, np.array([0.0, 1.0, 3.0]))


def test_get_dists_from_start_returns_nan_at_endpoints():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    path = Path(pts, 0.1)
    # Points exactly nearest to endpoints
    test_pts = np.array([[0.0, 0.0], [1.0, 2.0]])
    dists = path.get_dist_from_start(test_pts)
    assert np.isnan(dists[0])
    assert np.isnan(dists[1])


def test_get_dists_from_start_near_internal_vertices_and_edges():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    path = Path(pts, 0.25)
    # Points near x=0.9 (close to interior near index of ~0.9), and near x=2.2
    test_pts = np.array([[0.9, 0.01], [2.2, -0.02]])
    dists = path.get_dist_from_start(test_pts)
    # dists should be close to the along-path distances: ~0.9 and ~2.2
    assert dists[0] == pytest.approx(0.9, abs=0.2)
    assert dists[1] == pytest.approx(2.2, abs=0.2)


def test_get_dists_from_start_handles_many_points_vectorized():
    pts = np.array([[0.0, 0.0], [5.0, 0.0]])
    path = Path(pts, 0.1)
    xs = np.linspace(0.1, 4.9, 1000)
    test_pts = np.stack([xs, np.zeros_like(xs)], axis=1)
    dists = path.get_dist_from_start(test_pts)
    assert dists.shape == (1000,)
    # first and last are not endpoints; check monotonicity roughly
    assert np.all(np.diff(dists[np.isfinite(dists)]) >= 0)


def test_get_dists_from_start_2d_shape_check_and_type():
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    path = Path(pts, 0.5)
    test_pts = np.array([[0.5, 0.1], [1.5, -0.2]])
    dists = path.get_dist_from_start(test_pts)
    assert isinstance(dists, np.ndarray)
    assert dists.shape == (2,)


def test_get_dist_from_start_single_coord_returns_scalar_and_value():
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    path = Path(pts, 0.1)
    d = path.get_dist_from_start(np.array([0.75, 0.0]))
    assert np.isscalar(d)
    assert d == pytest.approx(0.75, abs=0.15)


def test_get_dist_from_start_single_coord_endpoint_nan():
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    path = Path(pts, 0.1)
    d_start = path.get_dist_from_start(np.array([0.0, 0.0]))
    d_end = path.get_dist_from_start(np.array([2.0, 0.0]))
    assert np.isnan(d_start)
    assert np.isnan(d_end)


def test_route_test_1():
    route = np.array(
        [
            [38.89210848595737, -77.03193105942654],
            [38.89209163234068, -77.02397135791017],
        ]
    )
    tp = np.array([[38.89209199365943, -77.0281295621989]])
    expected_dist = 1079  # ft from start, in route

    route = Route(route, distance_unit="feet", precision=1)
    dist = route.get_dist_from_start(tp)
    assert dist == pytest.approx(expected_dist, abs=1)


def test_route_test_2():
    route = np.array(
        [
            [34.14227811, -118.02865211],
            [34.14224251, -118.03154673],
            [34.14194166, -118.03312884],
            [34.14176138, -118.03350476],
            [34.14129116, -118.03409015],
            [34.14041136, -118.03492258],
            [34.13997281, -118.03530570],
            [34.13815629, -118.03705054],
            [34.13498725, -118.03994972],
        ]
    )

    tp = np.array(
        [
            [34.14063322, -118.03472264],
            [34.14218664, -118.03195910],
            [34.14242114, -118.02795183],
        ]
    )
    expected_dists = np.array([2048, 1000, np.nan])  # ft from start, in route

    route = Route(route, distance_unit="feet", precision=1)
    dist = route.get_dist_from_start(tp)
    # check one at a time
    assert dist[0] == pytest.approx(expected_dists[0], abs=1)
    assert dist[1] == pytest.approx(expected_dists[1], abs=1)
    assert np.isnan(dist[2])


def test_route_test_3():
    num_test_points = 1000000

    route = np.array([[0, 0], [1, 0]])
    tp = np.repeat(np.array([[0.5, 0.5]]), num_test_points, axis=0)
    route = Route(route, distance_unit="feet", precision=1)

    start_time = time.time()
    route.get_dist_from_start(tp)
    end_time = time.time()

    duration = end_time - start_time
    assert duration < 1
    print(f"Duration: {duration} seconds")
