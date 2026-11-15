"""Nominal racing path: one cubic Hermite curve per leg of the gate sequence.

Leg k ends in the center of gate ``gate_sequence[k]`` and starts in the center of the previous
gate in the sequence (leg 0 starts at the official takeoff position). The end tangents follow the
gates' passing normals, so the curve crosses each gate through its opening in the required
direction. The path is used for the start states between gates and for the arc-length progress
reward. All functions are batched over worlds and jittable.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.spatial.transform import Rotation

N_SEGMENTS = 32  # Polyline resolution for arc length and projection.


def passing_normals(gates_quat: Array, gate_ids: Array, direction: Array) -> Array:
    """Unit normals pointing in the required passing direction, shape (n, 3)."""
    rows = jnp.arange(gate_ids.shape[0])
    normal = Rotation.from_quat(gates_quat[rows, gate_ids]).apply(jnp.array([1.0, 0.0, 0.0]))
    return normal * direction[:, None]


def leg_control_points(
    gates_pos: Array,
    gates_quat: Array,
    gate_sequence: Array,
    gate_sequence_direction: Array,
    leg: Array,
    start_pos: Array,
) -> tuple[Array, Array, Array, Array]:
    """Hermite control points (p0, t0, p1, t1) of leg ``leg`` per world.

    Args:
        gates_pos: (n, n_gates, 3) gate centers.
        gates_quat: (n, n_gates, 4) gate orientations (xyzw).
        gate_sequence: (n_seq,) gate IDs in passing order.
        gate_sequence_direction: (n_seq,) +1/-1 passing direction per entry.
        leg: (n,) leg index in [0, n_seq); leg k ends at sequence entry k.
        start_pos: (3,) or (n, 3) official takeoff position, start of leg 0.
    """
    count = leg.shape[0]
    rows = jnp.arange(count)
    leg = jnp.clip(leg, 0, gate_sequence.shape[0] - 1)
    end_ids = gate_sequence[leg]
    prev_ids = gate_sequence[jnp.maximum(leg - 1, 0)]
    p1 = gates_pos[rows, end_ids]
    n1 = passing_normals(gates_quat, end_ids, gate_sequence_direction[leg])
    first = (leg == 0)[:, None]
    p0 = jnp.where(first, jnp.broadcast_to(start_pos, (count, 3)), gates_pos[rows, prev_ids])
    n0_gate = passing_normals(
        gates_quat, prev_ids, gate_sequence_direction[jnp.maximum(leg - 1, 0)]
    )
    chord = p1 - p0
    length = jnp.linalg.norm(chord, axis=-1, keepdims=True)
    n0 = jnp.where(first, chord / jnp.maximum(length, 1e-6), n0_gate)
    return p0, length * n0, p1, length * n1


def hermite(p0: Array, t0: Array, p1: Array, t1: Array, s: Array) -> tuple[Array, Array]:
    """Point and derivative d/ds of the cubic Hermite curve at s in [0, 1].

    ``s`` broadcasts against the leading axes of the control points: (n,) gives (n, 3),
    (n, m) gives (n, m, 3).
    """
    extra = s.ndim - p0.ndim + 1
    p0, t0, p1, t1 = (x.reshape(x.shape[:1] + (1,) * extra + x.shape[1:]) for x in (p0, t0, p1, t1))
    s = s[..., None]
    s2, s3 = s * s, s * s * s
    point = (
        (2 * s3 - 3 * s2 + 1) * p0
        + (s3 - 2 * s2 + s) * t0
        + (-2 * s3 + 3 * s2) * p1
        + (s3 - s2) * t1
    )
    tangent = (6 * s2 - 6 * s) * p0 + (3 * s2 - 4 * s + 1) * t0 + (6 * s - 6 * s2) * p1
    tangent = tangent + (3 * s2 - 2 * s) * t1
    return point, tangent


def polyline(p0: Array, t0: Array, p1: Array, t1: Array) -> tuple[Array, Array]:
    """Sample each leg as a polyline: points (n, N+1, 3) and cumulative arc length (n, N+1)."""
    s = jnp.broadcast_to(jnp.linspace(0.0, 1.0, N_SEGMENTS + 1), (p0.shape[0], N_SEGMENTS + 1))
    points, _ = hermite(p0, t0, p1, t1, s)
    seg = jnp.linalg.norm(jnp.diff(points, axis=1), axis=-1)
    cumulative = jnp.concatenate([jnp.zeros_like(seg[:, :1]), jnp.cumsum(seg, axis=1)], axis=1)
    return points, cumulative


def project(points: Array, cumulative: Array, pos: Array) -> tuple[Array, Array]:
    """Arc length of the closest point on the polyline, and the distance to it.

    Args:
        points: (n, N+1, 3) polyline vertices.
        cumulative: (n, N+1) arc length at each vertex.
        pos: (n, 3) query positions.
    """
    a, b = points[:, :-1], points[:, 1:]
    ab = b - a
    t = jnp.sum((pos[:, None] - a) * ab, axis=-1) / jnp.maximum(jnp.sum(ab * ab, axis=-1), 1e-12)
    t = jnp.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    dist = jnp.linalg.norm(pos[:, None] - closest, axis=-1)
    best = jnp.argmin(dist, axis=1)
    rows = jnp.arange(pos.shape[0])
    seg_len = cumulative[:, 1:] - cumulative[:, :-1]
    arc = cumulative[rows, best] + t[rows, best] * seg_len[rows, best]
    return arc, dist[rows, best]


def arc_progress(
    gates_pos: Array,
    gates_quat: Array,
    gate_sequence: Array,
    gate_sequence_direction: Array,
    leg: Array,
    start_pos: Array,
    prev_pos: Array,
    pos: Array,
    lateral_weight: float = 0.0,
) -> Array:
    """Change of the path potential arc(p) - lateral_weight * dist(p, path), shape (n,).

    Without the lateral term, any position beyond the gate's plane projects onto the end of
    the leg, so flying past the gate *beside* the opening would collect the full progress
    without passing the gate. Subtracting the distance to the path removes that shortcut; as a
    potential difference it only rewards changes, so a constant offset (e.g. a cut corner) costs
    nothing over time.
    """
    controls = leg_control_points(
        gates_pos, gates_quat, gate_sequence, gate_sequence_direction, leg, start_pos
    )
    points, cumulative = polyline(*controls)
    arc_before, dist_before = project(points, cumulative, prev_pos)
    arc_after, dist_after = project(points, cumulative, pos)
    return (arc_after - lateral_weight * dist_after) - (arc_before - lateral_weight * dist_before)
