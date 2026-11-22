"""Door models seen through pokes.

Positions and displacements live in the door frame: the plane of the
door, in metres, with the door spanning 0 to 1 m on both axes.
"""

import enum
from dataclasses import dataclass
from typing import Final

import numpy as np

from poke_or_ask.array_types import FloatArray, IntArray


class DoorKind(enum.IntEnum):
    """The three things a door can be. Values index probability arrays."""

    SLIDER = 0
    HINGE = 1
    LOCKED = 2


KIND_COUNT: Final = len(DoorKind)

# Length of one commanded push. A door with give fraction 1 moves by
# this distance along its free direction.
PUSH_DISTANCE_M: Final = 0.1
# A door with the smallest give moves at most 5 mm per poke, which is
# half of the default 1 cm sensor noise: the hard slider-versus-locked
# case.
GIVE_MIN_FRACTION: Final = 0.05
GIVE_MAX_FRACTION: Final = 1.0
# Hinge pivots are drawn from a square around the door, so some hinges
# sit inside the door and some well outside it.
PIVOT_MIN_M: Final = -0.5
PIVOT_MAX_M: Final = 1.5
# Scale of the lever factor r / (r + scale): motion vanishes at the
# pivot and approaches the full tangential push far from it.
HINGE_LEVER_SCALE_M: Final = 0.2
# Floor on the lever arm, so a poke exactly at the pivot does not divide
# by zero. The lever factor is zero there either way.
MIN_LEVER_M: Final = 1e-9
UNIT_NORM_TOLERANCE: Final = 1e-6


@dataclass(frozen=True, slots=True)
class HypothesisSet:
    """Doors described by a kind and hidden parameters, one per row.

    Parameters that do not apply to a kind are carried and ignored:
    a locked door uses neither the axis, the pivot, nor the give.

    Attributes
    ----------
    kind_index : shape (h,)
        A DoorKind value per row.
    axis_angle_rad : shape (h,)
        Free direction of a slider, measured from the door's x axis.
    pivot_door_m : shape (h, 2)
        Hinge pivot in the door frame.
    give_fraction : shape (h,)
        Fraction of the push that becomes motion.
    """

    kind_index: IntArray
    axis_angle_rad: FloatArray
    pivot_door_m: FloatArray
    give_fraction: FloatArray


def sample_hypotheses(count_per_kind: int, rng: np.random.Generator) -> HypothesisSet:
    """Draw doors from the prior, in one block per kind.

    Parameters
    ----------
    count_per_kind
        Rows drawn for each kind; the prior over kinds is uniform.
    rng
        Source of randomness.

    Returns
    -------
    hypotheses
        ``KIND_COUNT * count_per_kind`` rows, sliders first.
    """
    if count_per_kind < 1:
        raise ValueError(f"expected count_per_kind of at least 1, got {count_per_kind}")
    row_count = KIND_COUNT * count_per_kind
    return HypothesisSet(
        kind_index=np.repeat(np.arange(KIND_COUNT, dtype=np.int64), count_per_kind),
        axis_angle_rad=rng.uniform(0.0, np.pi, size=row_count),
        pivot_door_m=rng.uniform(PIVOT_MIN_M, PIVOT_MAX_M, size=(row_count, 2)),
        give_fraction=rng.uniform(GIVE_MIN_FRACTION, GIVE_MAX_FRACTION, size=row_count),
    )


def sample_doors(door_count: int, rng: np.random.Generator) -> HypothesisSet:
    """Draw doors whose kinds are uniform and independent.

    Returns
    -------
    doors
        ``door_count`` rows, each with fresh hidden parameters.
    """
    pool = sample_hypotheses(door_count, rng)
    kind_per_door = rng.integers(KIND_COUNT, size=door_count)
    row_per_door = kind_per_door * door_count + np.arange(door_count)
    return select_rows(pool, row_per_door)


def select_rows(hypotheses: HypothesisSet, row_indices: IntArray) -> HypothesisSet:
    """Return a copy holding only the listed rows."""
    return HypothesisSet(
        kind_index=hypotheses.kind_index[row_indices],
        axis_angle_rad=hypotheses.axis_angle_rad[row_indices],
        pivot_door_m=hypotheses.pivot_door_m[row_indices],
        give_fraction=hypotheses.give_fraction[row_indices],
    )


def noiseless_displacement_m(
    hypotheses: HypothesisSet,
    *,
    positions_door_m: FloatArray,
    push_directions: FloatArray,
) -> FloatArray:
    """Return how far each door moves under each poke, without noise.

    A slider moves only along its axis. A hinge moves tangentially to
    the circle around its pivot. A locked door does not move.

    Parameters
    ----------
    hypotheses
        The doors, ``h`` rows.
    positions_door_m : shape (c, 2)
        Where each poke lands.
    push_directions : shape (c, 2)
        Unit vectors, the push direction of each poke.

    Returns
    -------
    displacements_door_m : shape (c, h, 2)

    Raises
    ------
    ValueError
        If the shapes disagree or a push direction is not a unit vector.
    """
    _check_pokes(positions_door_m, push_directions)
    slider_motion = _slider_motion_fraction(hypotheses.axis_angle_rad, push_directions)
    hinge_motion = _hinge_motion_fraction(
        hypotheses.pivot_door_m, positions_door_m, push_directions
    )
    is_slider = (hypotheses.kind_index == DoorKind.SLIDER)[np.newaxis, :, np.newaxis]
    is_hinge = (hypotheses.kind_index == DoorKind.HINGE)[np.newaxis, :, np.newaxis]
    motion_fraction = np.where(is_slider, slider_motion, 0.0) + np.where(
        is_hinge, hinge_motion, 0.0
    )
    give_fraction = hypotheses.give_fraction[np.newaxis, :, np.newaxis]
    return np.asarray(PUSH_DISTANCE_M * give_fraction * motion_fraction, dtype=np.float64)


def observe_poke_m(
    door: HypothesisSet,
    *,
    position_door_m: FloatArray,
    push_direction: FloatArray,
    noise_std_m: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Return the displacement the robot measures for one poke.

    Parameters
    ----------
    door
        A single door, one row.
    position_door_m : shape (2,)
    push_direction : shape (2,)
    noise_std_m
        Standard deviation of the measurement noise on each axis.
    rng
        Source of randomness.

    Returns
    -------
    displacement_door_m : shape (2,)
    """
    if len(door.kind_index) != 1:
        raise ValueError(f"expected a single door, got {len(door.kind_index)} rows")
    if noise_std_m <= 0.0:
        raise ValueError(f"expected positive noise_std_m, got {noise_std_m}")
    noiseless_m = noiseless_displacement_m(
        door,
        positions_door_m=position_door_m[np.newaxis, :],
        push_directions=push_direction[np.newaxis, :],
    )[0, 0]
    return np.asarray(noiseless_m + rng.normal(0.0, noise_std_m, size=2), dtype=np.float64)


def _check_pokes(positions_door_m: FloatArray, push_directions: FloatArray) -> None:
    if positions_door_m.ndim != 2 or positions_door_m.shape[1] != 2:
        raise ValueError(f"expected positions of shape (c, 2), got {positions_door_m.shape}")
    if push_directions.shape != positions_door_m.shape:
        raise ValueError(
            f"expected push directions of shape {positions_door_m.shape}, "
            f"got {push_directions.shape}"
        )
    direction_norms = np.linalg.norm(push_directions, axis=1)
    if not np.allclose(direction_norms, 1.0, rtol=0.0, atol=UNIT_NORM_TOLERANCE):
        raise ValueError("expected unit push directions")


def _slider_motion_fraction(axis_angle_rad: FloatArray, push_directions: FloatArray) -> FloatArray:
    axes = np.stack([np.cos(axis_angle_rad), np.sin(axis_angle_rad)], axis=1)
    push_along_axis = push_directions @ axes.T
    return np.asarray(push_along_axis[..., np.newaxis] * axes[np.newaxis, :, :], dtype=np.float64)


def _hinge_motion_fraction(
    pivots_door_m: FloatArray, positions_door_m: FloatArray, push_directions: FloatArray
) -> FloatArray:
    offsets_m = positions_door_m[:, np.newaxis, :] - pivots_door_m[np.newaxis, :, :]
    lever_m = np.maximum(np.linalg.norm(offsets_m, axis=2), MIN_LEVER_M)
    tangents = np.stack([-offsets_m[..., 1], offsets_m[..., 0]], axis=2) / lever_m[..., np.newaxis]
    push_along_tangent = np.einsum("cd,chd->ch", push_directions, tangents)
    lever_fraction = lever_m / (lever_m + HINGE_LEVER_SCALE_M)
    return np.asarray(
        (push_along_tangent * lever_fraction)[..., np.newaxis] * tangents, dtype=np.float64
    )
