"""Unit tests for the path planner used by the MPCC controller."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.path_planning import ArcPath, PathConfig, PathPlanner, track_reference
from lsy_drone_racing.envs.utils import load_gate_order, load_track
from lsy_drone_racing.utils import load_config

CONFIG = load_config(Path(__file__).parents[3] / "config" / "level0.toml")


def make_obs(
    direction: np.ndarray | None = None, sequence: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Build a level-0 observation dictionary."""
    gates, obstacles, drones = load_track(CONFIG.env.track)
    order, sign = load_gate_order(CONFIG.env.track, len(CONFIG.env.track.gates))
    return {
        "pos": np.asarray(drones.pos[0], dtype=float),
        "vel": np.zeros(3),
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "gates_pos": np.asarray(gates.pos, dtype=float),
        "gates_quat": np.asarray(gates.quat, dtype=float),
        "obstacles_pos": np.asarray(obstacles.pos, dtype=float),
        "gate_sequence": order if sequence is None else sequence,
        "gate_sequence_direction": sign if direction is None else direction,
    }


@pytest.fixture(scope="module")
def planned() -> tuple:
    """Plan the level-0 path once for all tests."""
    obs = make_obs()
    planner = PathPlanner(PathConfig())
    path, info = planner.plan(obs, obs["pos"])
    return path, info, obs


def gate_crossings(
    positions: np.ndarray,
    gates_pos: np.ndarray,
    gates_quat: np.ndarray,
    sequence: np.ndarray,
    direction: np.ndarray,
    arc: np.ndarray,
) -> list[tuple[int, np.ndarray] | None]:
    """Return the arc length and in-plane offset of every configured gate crossing."""
    crossings: list[tuple[int, np.ndarray] | None] = []
    last = -1
    for gate_id, sign in zip(sequence, direction, strict=True):
        local = R.from_quat(gates_quat[gate_id]).apply(positions - gates_pos[gate_id], inverse=True)
        along = sign * local[:, 0]
        indices = np.where((along[:-1] < 0) & (along[1:] >= 0))[0]
        indices = indices[indices > last]
        if len(indices) == 0:
            crossings.append(None)
            continue
        i = int(indices[0])
        alpha = -local[i, 0] / (local[i + 1, 0] - local[i, 0])
        crossings.append((int(arc[i]), local[i, 1:] + alpha * (local[i + 1, 1:] - local[i, 1:])))
        last = i
    return crossings


@pytest.mark.unit
def test_gate_frames_match_configuration():
    """The gate normal must be the local x-axis of the gate rotation."""
    gates, _, _ = load_track(CONFIG.env.track)
    quats = np.asarray(gates.quat, dtype=float)
    normals = R.from_quat(quats).as_matrix()[:, :, 0]
    ypr = R.from_euler("xyz", np.asarray([g["rpy"] for g in CONFIG.env.track.gates], float))
    np.testing.assert_allclose(normals, ypr.apply(np.array([1.0, 0.0, 0.0])), atol=1e-6)


@pytest.mark.unit
def test_plan_follows_signed_gate_sequence(planned: tuple):
    """The path visits every entry of the gate order, including repeated and reverse passes."""
    _, info, obs = planned
    expected = obs["gates_pos"][obs["gate_sequence"]]
    np.testing.assert_allclose(info.gate_centers, expected)
    assert len(info.gate_centers) == len(obs["gate_sequence"])
    # Gate 1 is passed twice on level 0, so its centre must appear twice.
    repeated = np.all(np.isclose(info.gate_centers, obs["gates_pos"][1]), axis=1)
    assert repeated.sum() == 2


@pytest.mark.unit
def test_waypoints_respect_reverse_direction():
    """A reverse gate pass must be approached from the opposite side of the gate."""
    obs = make_obs(direction=np.array([1, -1, 1, 1]), sequence=np.array([0, 1, 2, 3]))
    planner = PathPlanner(PathConfig())
    path, _ = planner.plan(obs, obs["pos"])
    arc = np.linspace(0.0, path.length, 8000)
    positions = path.pos(arc)
    tangents = path.tangent(arc)
    for gate_id, sign in zip(obs["gate_sequence"], obs["gate_sequence_direction"], strict=True):
        rotation = R.from_quat(obs["gates_quat"][gate_id])
        # The path has to pass through the gate centre and do so along the configured direction.
        nearest = int(np.argmin(np.linalg.norm(positions - obs["gates_pos"][gate_id], axis=1)))
        assert np.linalg.norm(positions[nearest] - obs["gates_pos"][gate_id]) < 0.15
        tangent_local = rotation.apply(tangents[nearest], inverse=True)
        assert np.sign(tangent_local[0]) == np.sign(sign), "gate direction is not respected"


@pytest.mark.unit
def test_path_is_continuous_and_arc_length_parameterized(planned: tuple):
    """The path must be continuous, unit-tangent and parameterized by arc length."""
    path, _, obs = planned
    arc = np.linspace(0.0, path.length, 4000)
    positions = path.pos(arc)
    np.testing.assert_allclose(positions[0], obs["pos"], atol=1e-6)
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    assert abs(segment_lengths.sum() - path.length) / path.length < 0.02
    tangents = path.tangent(arc)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0, atol=1e-6)
    # The linear extension beyond the path must be continuous.
    beyond = path.pos(np.array([path.length, path.length + 0.1]))
    np.testing.assert_allclose(beyond[1] - beyond[0], 0.1 * path.tangent(path.length), atol=1e-6)


@pytest.mark.unit
def test_path_keeps_clearance_and_stays_above_ground(planned: tuple):
    """Obstacle avoidance and the take-off waypoint must both hold."""
    path, info, _ = planned
    positions = path.pos(np.linspace(0.0, path.length, 6000))
    assert PathPlanner.clearance(positions, info.obstacles) >= 0.27
    assert positions[:, 2].min() > 0.0


@pytest.mark.unit
def test_gate_crossings_are_centred(planned: tuple):
    """Every gate must be crossed close to its centre and in the configured direction."""
    path, info, obs = planned
    arc = np.linspace(0.0, path.length, 20000)
    crossings = gate_crossings(
        path.pos(arc),
        obs["gates_pos"],
        obs["gates_quat"],
        obs["gate_sequence"],
        obs["gate_sequence_direction"],
        arc,
    )
    assert all(crossing is not None for crossing in crossings), "a gate is never crossed"
    offsets = np.array([crossing[1] for crossing in crossings])
    assert np.all(np.abs(offsets) < 0.1), f"gate offsets {offsets} are not centred"
    arc_indices = [crossing[0] for crossing in crossings]
    assert arc_indices == sorted(arc_indices), "gate passes are not in the configured order"
    assert info.gate_centers.shape[0] == len(crossings)


@pytest.mark.unit
def test_closest_projection(planned: tuple):
    """The projection recovers the arc length of points on the path and never runs backwards."""
    path, _, _ = planned
    for theta in (0.5, 3.0, 7.0, path.length - 0.5):
        projected = path.closest(path.pos(theta), theta)
        assert abs(projected - theta) < 0.05
    # A point far behind the current progress must not drag the projection back to the path start.
    projected = path.closest(path.pos(0.2), 4.0)
    assert 4.0 - 0.15 - 1e-6 <= projected <= 4.6


@pytest.mark.unit
def test_path_curvature_is_flyable(planned: tuple):
    """The path must be smooth overall; local corners are handled by the curvature speed limit."""
    path, _, _ = planned
    curvature = path.curvature(np.linspace(0.0, path.length, 1000))
    assert np.all(np.isfinite(curvature))
    assert np.all(curvature >= 0.0)
    # The blend between the straightened gate passage and the rest of the path is a local corner.
    # It is bounded, and MPCCConfig.a_lat_max slows the reference down there, so the requirement is
    # that it stays finite and that the path is smooth on average.
    assert curvature.max() < 30.0
    assert curvature.mean() < 4.0
    # A straight path has no curvature.
    straight = ArcPath(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]))
    assert float(np.max(straight.curvature(np.linspace(0.0, straight.length, 50)))) < 1e-6


@pytest.mark.unit
def test_progress_weight_peaks_at_gates(planned: tuple):
    """The contouring weight must be high at the gates and decay in between."""
    path, info, _ = planned
    weights = track_reference(path, np.array([0.0]), info.gate_centers, info.obstacles)[2]
    assert weights.shape == (1,)
    samples = path.pos(np.linspace(0, path.length, 4000))
    gate_theta = [
        float(np.argmin(np.linalg.norm(samples - centre, axis=1))) * path.length / 3999
        for centre in info.gate_centers
    ]
    at_gates = track_reference(path, np.array(gate_theta), info.gate_centers, info.obstacles)[2]
    assert np.all(at_gates > 0.3)
    away = track_reference(path, np.array([path.length]), info.gate_centers, info.obstacles)[2]
    assert away[0] < 0.3
