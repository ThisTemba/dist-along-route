# dist-along-route

A Python library for computing distances along a route from latitude/longitude coordinates. This library efficiently calculates how far along a route any given point is. It is best used for calculating how far along transit vehicles are on their routes, but can be applied to any scenario where you need to measure distances along a predefined path.

## Installation

```bash
pip install dist-along-route
```

## Quick Start

```python
import numpy as np

from dist_along_route import Route

# Define a route as lat/lon coordinates (in this case the path from the Lincoln Memorial to the US Capitol)
route_coords = np.array(
    [
        [38.88927893, -77.05017266],
        [38.88981981, -77.00906834],
    ]
)

# Create a Route object
route = Route(route_coords, distance_unit="feet", precision=1)

# Test point (lat/lon)
p1 = np.array([38.89767634, -77.03653007])  # White House
p2 = np.array([38.88946913, -77.03524507])  # Washington Monument
test_points = np.array([p1, p2])

# Get distance from start of route
distances_from_start, perpendicular_distances = route.get_dist_from_start(test_points)

print(distances_from_start)
print(perpendicular_distances)
```

## How it Works

### Basics

1. **Route Initialization**: The `Route` class takes an array of latitude/longitude points defining the route. These are first converted to Cartesian coordinates for accurate distance calculations.
2. **Path Subdivision**: The route is subdivided into smaller segments such that no segment is larger than the specified precision.
3. **Distance Calculation**: For each point in the route, the library computes the distance of that point from the start of the route.
4. **KDTree Search**: A KDTree is built from the subdivided route points to allow for efficient nearest neighbor searches.
5. **Point Processing**: The `get_dist_from_start` method takes in test points, finds the nearest neighbor using the KDTree, then looks up the distance of that route point from the start of the route.

### Subtleties

1. **Precision Control**: The `precision` parameter controls how finely the route is subdivided. A smaller precision leads to more accurate distance calculations but increases computation time.
2. **Distance Units**: The library supports four distance units: meters, feet, miles, and kilometers. The choice of unit affects the output distances. This unit applies to the precision, distances from the start, and perpendicular distances.
3. **Off-Route Points**: Test points whose nearest neighbor is the first or last point of the route are considered off-route and will return NaN for their distance from the start.
4. **Perpendicular Distance Filtering**: The optional `max_dist_from_route` parameter allows users to classify points that are too far from the route as off-route, returning NaN for their distance from the start.

## API Reference

### Route Class

The main class for route distance calculations.

#### Constructor

```python
Route(lat_lon_pts, precision=1.0, distance_unit="feet")
```

**Parameters:**

- `lat_lon_pts` (np.ndarray): Array of shape (n, 2) containing lat/lon coordinates
- `precision` (float): Maximum distance between subdivided path points (default: 1.0)
- `distance_unit` (str): Unit for distance calculations - "meters", "feet", "miles", or "kilometers" (default: "feet")

#### Methods

##### get_dist_from_start()

```python
get_dist_from_start(lat_lon_pt, max_dist_from_route=None)
```

**Parameters:**

- `lat_lon_pt` (np.ndarray): Single point as [lat, lon] or array of points with shape (n, 2)
- `max_dist_from_route` (float, optional): Maximum perpendicular distance from route. Points beyond this distance will return NaN.

**Returns:**

- Tuple of (distances_from_start, distances_to_route) in the same unit as the route

## Distance Units

Supported distance units:

- `"meters"` - SI base unit
- `"feet"` - US customary unit (default)
- `"miles"` - US customary unit
- `"kilometers"` - SI unit

## Precision Control

The `precision` parameter controls the accuracy of distance calculations:

- **Lower values** (e.g., 0.1) = Higher accuracy, slower processing
- **Higher values** (e.g., 10.0) = Lower accuracy, faster processing

Choose based on your accuracy requirements and performance needs.
