import time
from typing import Any, cast

import numpy as np
import pytest

from dist_along_route import Route
from dist_along_route.path import _Path as Path
from dist_along_route.path import subdivide_path


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
    with pytest.raises(ValueError):
        Path(cast(Any, [[0.0, 0.0], [1.0, 0.0]]), 0.1)  # not np.ndarray
    with pytest.raises(ValueError):
        Path(np.array([[0.0, 0.0]]), 0.1)  # less than 2 points
    with pytest.raises(ValueError):
        Path(np.array([[0.0], [1.0]]), 0.1)  # not 2D
    with pytest.raises(ValueError):
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
    dists, perp_dists = path.get_dist_from_start(test_pts)
    assert np.isnan(dists[0])
    assert np.isnan(dists[1])


def test_get_dists_from_start_near_internal_vertices_and_edges():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    path = Path(pts, 0.25)
    # Points near x=0.9 (close to interior near index of ~0.9), and near x=2.2
    test_pts = np.array([[0.9, 0.01], [2.2, -0.02]])
    dists, perp_dists = path.get_dist_from_start(test_pts)
    # dists should be close to the along-path distances: ~0.9 and ~2.2
    assert dists[0] == pytest.approx(0.9, abs=0.2)
    assert dists[1] == pytest.approx(2.2, abs=0.2)


def test_get_dists_from_start_handles_many_points_vectorized():
    pts = np.array([[0.0, 0.0], [5.0, 0.0]])
    path = Path(pts, 0.1)
    xs = np.linspace(0.1, 4.9, 1000)
    test_pts = np.stack([xs, np.zeros_like(xs)], axis=1)
    dists, perp_dists = path.get_dist_from_start(test_pts)
    assert dists.shape == (1000,)
    # first and last are not endpoints; check monotonicity roughly
    assert np.all(np.diff(dists[np.isfinite(dists)]) >= 0)


def test_get_dists_from_start_2d_shape_check_and_type():
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    path = Path(pts, 0.5)
    test_pts = np.array([[0.5, 0.1], [1.5, -0.2]])
    dists, perp_dists = path.get_dist_from_start(test_pts)
    assert isinstance(dists, np.ndarray)
    assert dists.shape == (2,)


def test_get_dist_from_start_single_coord_returns_scalar_and_value():
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    path = Path(pts, 0.1)
    dists, perp_dists = path.get_dist_from_start(np.array([0.75, 0.0]))
    assert np.isscalar(dists)
    assert dists == pytest.approx(0.75, abs=0.15)


def test_get_dist_from_start_single_coord_endpoint_nan():
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    path = Path(pts, 0.1)
    dists, perp_dists = path.get_dist_from_start(np.array([0.0, 0.0]))
    assert np.isnan(dists)
    dists, perp_dists = path.get_dist_from_start(np.array([2.0, 0.0]))
    assert np.isnan(dists)


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
    dists, perp_dists = route.get_dist_from_start(tp)
    assert dists == pytest.approx(expected_dist, abs=1)


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
    dists, perp_dists = route.get_dist_from_start(tp)
    # check one at a time
    assert isinstance(dists, np.ndarray)
    assert dists[0] == pytest.approx(expected_dists[0], abs=1)
    assert dists[1] == pytest.approx(expected_dists[1], abs=1)
    assert np.isnan(dists[2])


def test_route_test_3():
    num_test_points = 1000000

    route = np.array([[0, 0], [1, 0]])
    tp = np.repeat(np.array([[0.5, 0.5]]), num_test_points, axis=0)
    route = Route(route, distance_unit="feet", precision=1)

    start_time = time.time()
    route.get_dist_from_start(tp)
    end_time = time.time()

    duration = end_time - start_time
    assert duration < 5
    print(f"Duration: {duration} seconds")


def test_perpendicular_distance_basic():
    """Test basic perpendicular distance calculation for a simple horizontal line."""
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    path = Path(pts, 0.1)

    # Test point directly above the line
    test_pt = np.array([1.0, 1.0])
    dists, perp_dists = path.get_dist_from_start(test_pt)

    assert dists == pytest.approx(1.0, abs=0.1)  # distance along path
    assert perp_dists == pytest.approx(1.0, abs=0.1)  # perpendicular distance


def test_perpendicular_distance_below_line():
    """Test perpendicular distance for point below the line."""
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    path = Path(pts, 0.1)

    # Test point directly below the line
    test_pt = np.array([1.0, -0.5])
    dists, perp_dists = path.get_dist_from_start(test_pt)

    assert dists == pytest.approx(1.0, abs=0.1)  # distance along path
    assert perp_dists == pytest.approx(0.5, abs=0.1)  # perpendicular distance


def test_perpendicular_distance_vertical_line():
    """Test perpendicular distance for a vertical line."""
    pts = np.array([[0.0, 0.0], [0.0, 2.0]])
    path = Path(pts, 0.1)

    # Test point to the right of the line
    test_pt = np.array([1.0, 1.0])
    dists, perp_dists = path.get_dist_from_start(test_pt)

    assert dists == pytest.approx(1.0, abs=0.1)  # distance along path
    assert perp_dists == pytest.approx(1.0, abs=0.1)  # perpendicular distance


def test_perpendicular_distance_diagonal_line():
    """Test perpendicular distance for a diagonal line."""
    # Using a 3-4-5 triangle for nice integer values
    pts = np.array([[0.0, 0.0], [4.0, 3.0]])  # diagonal line with slope 3/4
    path = Path(pts, 0.1)

    # Test point that forms a 3-4-5 triangle with the path
    test_pt = np.array([4.0, 0.0])  # point directly below the end
    dists, perp_dists = path.get_dist_from_start(test_pt)

    assert dists == pytest.approx(3.2, abs=0.01)  # distance along path to closest point
    assert perp_dists == pytest.approx(2.4, abs=0.01)  # perpendicular distance (height)


def test_perpendicular_distance_multiple_points():
    """Test perpendicular distance calculation for multiple points."""
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    path = Path(pts, 0.1)

    # Multiple test points at different perpendicular distances
    test_pts = np.array(
        [
            [0.5, 0.5],  # 0.5 units above
            [1.0, 1.0],  # 1.0 units above
            [1.5, -0.3],  # 0.3 units below
        ]
    )

    dists, perp_dists = path.get_dist_from_start(test_pts)

    assert dists.shape == (3,)
    assert perp_dists.shape == (3,)

    # Check along-path distances
    assert dists[0] == pytest.approx(0.5, abs=0.1)
    assert dists[1] == pytest.approx(1.0, abs=0.1)
    assert dists[2] == pytest.approx(1.5, abs=0.1)

    # Check perpendicular distances
    assert perp_dists[0] == pytest.approx(0.5, abs=0.1)
    assert perp_dists[1] == pytest.approx(1.0, abs=0.1)
    assert perp_dists[2] == pytest.approx(0.3, abs=0.1)


def test_perpendicular_distance_complex_path():
    """Test perpendicular distance for a complex path with multiple segments."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    path = Path(pts, 0.1)

    # Test points near different segments
    test_pts = np.array(
        [
            [0.5, 0.2],  # Above first horizontal segment
            [1.0, 0.5],  # Near vertical segment (exactly on it)
            [1.5, 0.8],  # Below second horizontal segment
        ]
    )

    dists, perp_dists = path.get_dist_from_start(test_pts)

    assert dists.shape == (3,)
    assert perp_dists.shape == (3,)

    # Check each perpendicular distance individually
    assert perp_dists[0] == pytest.approx(0.2, abs=0.1)  # Above first segment
    assert perp_dists[1] == pytest.approx(0.0, abs=0.1)  # Exactly on vertical segment
    assert perp_dists[2] == pytest.approx(0.2, abs=0.1)  # Below second segment


def test_perpendicular_distance_endpoints():
    """Test that perpendicular distances work correctly near endpoints."""
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    path = Path(pts, 0.1)

    # Test points near endpoints
    test_pts = np.array(
        [
            [-1, 0.5],  # Near start, above
            [3, -0.3],  # Near end, below
        ]
    )

    dists, perp_dists = path.get_dist_from_start(test_pts)

    # Along-path distances should be nan for points near endpoints
    assert np.isnan(dists[0])
    assert np.isnan(dists[1])

    # Perpendicular distances should be nan for points near endpoints
    assert np.isnan(perp_dists[0])
    assert np.isnan(perp_dists[1])


def test_perpendicular_distance_route_integration():
    """Test perpendicular distance calculation with Route class."""
    # Use small lat/lon values to avoid large coordinate conversion issues
    route = np.array(
        [[39.95172763, -75.15831882], [39.95114298, -75.15359534]]
    )  # Very small lat/lon change
    route_obj = Route(route, distance_unit="feet", precision=0.5)

    # Test point above the route
    test_pt = np.array([[39.95073982, -75.15690203]])  # Halfway along, slightly above
    dists, perp_dists = route_obj.get_dist_from_start(test_pt)
    # expect perp distance to be 292 ft +- 1 ft
    assert perp_dists == pytest.approx(292, abs=1)


def test_perpendicular_distance_precision_effect():
    """Test that perpendicular distance accuracy depends on path precision."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])

    # Test with different precision values
    for precision in [0.5, 0.1, 0.01]:
        path = Path(pts, precision)
        test_pt = np.array([0.5, 0.5])
        dists, perp_dists = path.get_dist_from_start(test_pt)

        # Perpendicular distance should be close to 0.5 regardless of precision
        assert perp_dists == pytest.approx(0.5, abs=precision)


def test_perpendicular_distance_edge_cases():
    """Test perpendicular distance for edge cases."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    path = Path(pts, 0.1)

    # Test point very close to the path
    test_pt = np.array([0.5, 0.001])
    dists, perp_dists = path.get_dist_from_start(test_pt)

    assert dists == pytest.approx(0.5, abs=0.1)
    assert perp_dists == pytest.approx(0.001, abs=0.1)

    # Test point very far from the path
    test_pt = np.array([0.5, 100.0])
    dists, perp_dists = path.get_dist_from_start(test_pt)

    assert dists == pytest.approx(0.5, abs=0.1)
    assert perp_dists == pytest.approx(100.0, abs=0.1)


def test_max_dist_from_route_basic():
    """Test basic max_dist_from_route functionality."""
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    path = Path(pts, 0.1)

    # Test points at different distances from the route
    test_pts = np.array(
        [
            [0.5, 0.1],  # Close to route (0.1 units away)
            [1.0, 0.5],  # Medium distance (0.5 units away)
            [1.5, 1.0],  # Far from route (1.0 units away)
        ]
    )

    # Without max_dist_from_route - all points should return values
    dists, perp_dists = path.get_dist_from_start(test_pts)
    assert np.all(np.isfinite(dists))
    assert np.all(np.isfinite(perp_dists))

    # With max_dist_from_route = 0.3 - only first point should return values
    dists, perp_dists = path.get_dist_from_start(test_pts, max_dist_from_route=0.3)
    assert np.isfinite(dists[0])  # Close point
    assert np.isnan(dists[1])  # Medium distance point
    assert np.isnan(dists[2])  # Far point

    assert np.isfinite(perp_dists[0])  # Close point
    assert np.isnan(perp_dists[1])  # Medium distance point
    assert np.isnan(perp_dists[2])  # Far point


def test_max_dist_from_route_route_integration():
    """Test max_dist_from_route with Route class."""
    route = np.array([[0.0, 0.0], [0.001, 0.0]])  # Small lat/lon change
    route_obj = Route(route, distance_unit="meters", precision=0.1)

    # Test points at different distances
    test_pts = np.array(
        [
            [0.0005, 0.0001],  # Close to route
            [0.0005, 0.0010],  # Far from route
        ]
    )

    # With max_dist_from_route
    dists, perp_dists = route_obj.get_dist_from_start(
        test_pts, max_dist_from_route=50
    )  # 50 meters

    # First point should be included (close)
    assert np.isfinite(dists[0])
    assert np.isfinite(perp_dists[0])

    # Second point should be excluded (far)
    assert np.isnan(dists[1])
    assert np.isnan(perp_dists[1])


def main():
    test_get_dist_from_start_single_coord_endpoint_nan()
    test_get_dist_from_start_single_coord_returns_scalar_and_value()
    test_get_dists_from_start_near_internal_vertices_and_edges()
    test_get_dists_from_start_returns_nan_at_endpoints()
    test_get_path_dists_from_start_basic()
    test_path_init_validation_errors()
    test_perpendicular_distance_basic()
    test_perpendicular_distance_below_line()
    test_perpendicular_distance_vertical_line()
    test_perpendicular_distance_diagonal_line()
    test_perpendicular_distance_multiple_points()
    test_perpendicular_distance_complex_path()
    test_perpendicular_distance_endpoints()
    test_perpendicular_distance_route_integration()
    test_perpendicular_distance_precision_effect()
    test_perpendicular_distance_edge_cases()
    test_route_test_3()
    test_route_test_1()
    test_route_test_2()
    test_subdivide_path_no_subdivision_when_within_max_dist()
    test_subdivide_path_single_segment_exact_division()
    test_subdivide_path_single_segment_non_exact_division()
    test_subdivide_path_multi_segment_mixed()
    test_path_init_validation_errors()
    test_get_dists_from_start_handles_many_points_vectorized()
    test_get_dists_from_start_2d_shape_check_and_type()
    test_max_dist_from_route_basic()
    test_max_dist_from_route_route_integration()


if __name__ == "__main__":
    main()
