import numpy as np


def heart_shape(count=9000):
    t = np.random.uniform(0, 2 * np.pi, count)

    x = 16 * np.sin(t) ** 3
    y = (
        13 * np.cos(t)
        - 5 * np.cos(2 * t)
        - 2 * np.cos(3 * t)
        - np.cos(4 * t)
    )

    scale = np.sqrt(np.random.uniform(0.15, 1.0, count))

    x *= scale / 18
    y *= scale / 18

    z = np.random.normal(0, 0.10, count)

    return np.column_stack((x, y, z))


def galaxy_shape(count=9000):
    arms = np.random.randint(0, 5, count)

    radius = np.sqrt(np.random.uniform(0.01, 1.0, count))
    angle = radius * 9 + arms * (2 * np.pi / 5)

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    z = np.random.normal(0, 0.08, count)

    # slight galaxy thickness
    z += radius * 0.12

    return np.column_stack((x, y, z))


def cloud_shape(count=9000):
    # Random volumetric nebula
    points = np.random.normal(0, 0.35, (count, 3))

    distance = np.linalg.norm(points, axis=1)

    mask = distance > 1.0

    while np.any(mask):
        points[mask] = np.random.normal(
            0, 0.35, (np.sum(mask), 3)
        )
        distance = np.linalg.norm(points, axis=1)
        mask = distance > 1.0

    # distort cloud
    points[:, 0] *= 1.25
    points[:, 1] *= 0.85
    points[:, 2] *= 0.65

    return points


def star_burst_shape(count=9000):
    direction = np.random.normal(0, 1, (count, 3))
    direction /= np.linalg.norm(
        direction, axis=1, keepdims=True
    )

    radius = np.random.power(4, count) * 1.2

    points = direction * radius[:, None]

    # stronger central core
    core = np.random.random(count) < 0.25
    points[core] *= 0.35

    return points


def generate_all_shapes(count=9000):

    return {
        "heart": heart_shape(count),
        "galaxy": galaxy_shape(count),
        "cloud": cloud_shape(count),
        "star": star_burst_shape(count),
    }