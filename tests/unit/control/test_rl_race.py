"""Pure functions of the Brax racing pipeline: path, observations, actions, rewards."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.rl_race.env_wrapper import path, rewards
from lsy_drone_racing.control.rl_race.env_wrapper.actions import ActionConfig, to_command
from lsy_drone_racing.control.rl_race.env_wrapper.observations import ObsConfig, encode

# Two gates on the x axis, both passed in +x, plus one reversed entry.
GATES_POS = jnp.array([[[1.0, 0.0, 1.0], [3.0, 0.0, 1.0]]])
GATES_QUAT = jnp.array([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])
SEQUENCE = jnp.array([0, 1])
DIRECTION = jnp.array([1.0, 1.0])
START = jnp.array([-1.0, 0.0, 1.0])


def test_hermite_hits_control_points_and_tangents():
    p0, t0, p1, t1 = (
        jnp.array([[0.0, 0, 0]]),
        jnp.array([[1.0, 2, 0]]),
        jnp.array([[2.0, 0, 1]]),
        jnp.array([[0.0, 1, 1]]),
    )
    for s, point, tangent in ((0.0, p0, t0), (1.0, p1, t1)):
        got_point, got_tangent = path.hermite(p0, t0, p1, t1, jnp.array([s]))
        np.testing.assert_allclose(got_point, point, atol=1e-6)
        np.testing.assert_allclose(got_tangent, tangent, atol=1e-6)


def test_leg_controls_use_start_then_previous_gate():
    for leg, p0 in ((0, START), (1, GATES_POS[0, 0])):
        c = path.leg_control_points(
            GATES_POS, GATES_QUAT, SEQUENCE, DIRECTION, jnp.array([leg]), START
        )
        np.testing.assert_allclose(c[0][0], p0, atol=1e-6)
        np.testing.assert_allclose(c[2][0], GATES_POS[0, leg], atol=1e-6)
        # Straight course along +x: both tangents point along +x with the chord length.
        np.testing.assert_allclose(c[1][0], [2.0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(c[3][0], [2.0, 0, 0], atol=1e-6)


def test_straight_leg_arc_length_projection_and_progress():
    controls = path.leg_control_points(
        GATES_POS, GATES_QUAT, SEQUENCE, DIRECTION, jnp.array([1]), START
    )
    points, cumulative = path.polyline(*controls)
    assert float(cumulative[0, -1]) == pytest.approx(2.0, rel=1e-4)
    arc, dist = path.project(points, cumulative, jnp.array([[2.0, 0.3, 1.0]]))
    assert float(arc[0]) == pytest.approx(1.0, abs=1e-3)
    assert float(dist[0]) == pytest.approx(0.3, abs=1e-3)
    gain = path.arc_progress(
        GATES_POS,
        GATES_QUAT,
        SEQUENCE,
        DIRECTION,
        jnp.array([1]),
        START,
        jnp.array([[1.5, 0, 1.0]]),
        jnp.array([[1.7, 0.1, 1.0]]),
    )
    assert float(gain[0]) == pytest.approx(0.2, abs=1e-3)


def test_reversed_gate_flips_the_passing_normal():
    quats = jnp.repeat(GATES_QUAT, 2, axis=0)  # two worlds
    normal = path.passing_normals(quats, jnp.array([0, 1]), jnp.array([1.0, -1.0]))
    np.testing.assert_allclose(normal, [[1, 0, 0], [-1, 0, 0]], atol=1e-6)


def _raw(pos: list[float], yaw: float = 0.0, passed: int = 0) -> dict[str, jnp.ndarray]:
    quat = R.from_euler("z", yaw).as_quat()
    return {
        "pos": jnp.array([pos]),
        "quat": jnp.array([quat]),
        "vel": jnp.array([[1.0, 0, 0]]),
        "ang_vel": jnp.zeros((1, 3)),
        "n_gates_passed": jnp.array([passed]),
        "gate_sequence": SEQUENCE[None],
        "gate_sequence_direction": DIRECTION[None],
        "gates_pos": GATES_POS,
        "gates_quat": GATES_QUAT,
        "obstacles_pos": jnp.array([[[0.0, 3.0, 1.5], [0.5, 0.2, 1.5]]]),
    }


def test_observation_layout_body_frame_and_padding():
    cfg = ObsConfig(n_gates=2, n_obstacles=2)
    prev = jnp.zeros((1, 4))
    level = np.asarray(encode(_raw([0.0, 0, 1.0]), prev, cfg))[0]
    assert level.shape == (cfg.size,)
    np.testing.assert_allclose(level[23:26], [1.0, 0, 0], atol=1e-6)  # gate 0 center, relative
    np.testing.assert_allclose(level[26:29], [1.0, 0, 0], atol=1e-6)  # gate 0 normal
    assert level[29] == 1.0  # valid
    np.testing.assert_allclose(level[37:40], [0.5, 0.2, 0.5], atol=1e-6)  # nearest obstacle first
    # Yawed by +90 degrees, world +x appears as body -y.
    yawed = np.asarray(encode(_raw([0.0, 0, 1.0], yaw=np.pi / 2), prev, cfg))[0]
    np.testing.assert_allclose(yawed[23:26], [0, -1.0, 0], atol=1e-6)
    # After the last gate of the sequence, the second slot is zero padding.
    last = np.asarray(encode(_raw([2.0, 0, 1.0], passed=1), prev, cfg))[0]
    assert last[29] == 1.0 and np.all(last[30:37] == 0)


def test_action_mapping_limits_and_yaw():
    cfg = ActionConfig(roll_limit_deg=70.0, thrust_min=0.1, thrust_max=0.8)
    cmd = np.asarray(to_command(jnp.array([[1.0, -1.0, 0.5, -1.0], [0.0, 0.5, -1.0, 1.0]]), cfg))
    np.testing.assert_allclose(
        cmd[0], [np.deg2rad(70), -np.deg2rad(70), np.deg2rad(35), 0.1], atol=1e-6
    )
    np.testing.assert_allclose(cmd[1], [0.0, np.deg2rad(35), -np.deg2rad(70), 0.8], atol=1e-6)
    with pytest.raises(ValueError):
        ActionConfig(yaw_limit_deg=95)


def test_reward_events_and_components():
    prev = jnp.array([0, 1, 4, 2])
    passed = jnp.array([1, 1, 5, 3])
    terminated = jnp.array([False, True, False, True])
    ev = rewards.events(prev, passed, terminated, n_seq=5)
    np.testing.assert_array_equal(ev["gates"], [1, 0, 1, 0])  # crash step pays no gate
    np.testing.assert_array_equal(ev["finished"], [False, False, True, False])
    np.testing.assert_array_equal(ev["crashed"], [False, True, False, True])
    cfg = rewards.preset("dense_path")
    total, comp = rewards.compute(cfg, ev, jnp.array([[0, 0, 2.0]] * 4), jnp.array([0.1] * 4))
    np.testing.assert_allclose(comp["progress"], [1.5, 0.0, 1.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(comp["ang_vel"], [-0.01] * 4, atol=1e-7)
    np.testing.assert_allclose(total[2], 10 + 50 - 0.05 - 0.01 + 1.5, atol=1e-5)
    np.testing.assert_allclose(total[1], -20 - 0.05 - 0.01, atol=1e-5)
    assert rewards.preset("race").progress == 0.0
    np.testing.assert_allclose(comp["lap_bonus"], 0.0)  # default: no lap bonus


def test_lap_bonus_pays_only_on_the_finishing_step():
    ev = rewards.events(jnp.array([4, 4, 2]), jnp.array([5, 5, 3]), jnp.zeros(3, bool), n_seq=5)
    cfg = rewards.preset("race", lap_bonus=20.0, lap_ref=4.0)
    seconds = jnp.array([3.5, 4.2, 1.0])
    _, comp = rewards.compute(cfg, ev, jnp.zeros((3, 3)), jnp.zeros(3), seconds)
    np.testing.assert_allclose(comp["lap_bonus"], [10.0, -4.0, 0.0], atol=1e-5)


def test_path_potential_penalizes_passing_beside_the_gate():
    args = (GATES_POS, GATES_QUAT, SEQUENCE, DIRECTION, jnp.array([1]), START)
    before = jnp.array([[2.5, 0.0, 1.0]])  # on the path, 0.5 m before gate 1 (x = 3)
    through = jnp.array([[3.2, 0.0, 1.0]])  # beyond the plane, through the opening
    beside = jnp.array([[3.2, 1.4, 1.0]])  # beyond the plane, 1.4 m beside the opening
    plain = [float(path.arc_progress(*args, before, p)[0]) for p in (through, beside)]
    assert plain[0] == pytest.approx(plain[1], abs=1e-3)  # arc alone cannot tell them apart
    shaped = [
        float(path.arc_progress(*args, before, p, lateral_weight=1.0)[0]) for p in (through, beside)
    ]
    assert shaped[0] - shaped[1] == pytest.approx(np.hypot(0.2, 1.4) - 0.2, abs=1e-2)
    # A constant offset parallel to the path costs nothing.
    side_a, side_b = jnp.array([[1.8, 0.3, 1.0]]), jnp.array([[2.2, 0.3, 1.0]])
    gain = float(path.arc_progress(*args, side_a, side_b, lateral_weight=1.0)[0])
    assert gain == pytest.approx(0.4, abs=1e-2)


def test_noise_scale_scales_only_the_sampling_std():
    from lsy_drone_racing.control.rl_race.network import make_networks

    params = jnp.concatenate([jnp.zeros(3), jnp.full(3, 0.5)])[None]
    stds = [
        make_networks(4, 3, noise_scale=s, distribution_type="tanh_normal")
        .parametric_action_distribution.create_dist(params)
        .scale
        for s in (1.0, 0.5)
    ]
    np.testing.assert_allclose(stds[1], 0.5 * stds[0], rtol=1e-6)


def test_lr_schedule_freeze_warmup_decay():
    from lsy_drone_racing.control.rl_race.optim import lr_schedule

    s = lr_schedule(1.0, freeze_steps=10, warmup_steps=10, decay_steps=100, final_lr_factor=0.1)
    np.testing.assert_allclose(
        [float(s(t)) for t in (0, 9, 15, 20, 70, 120, 500)],
        [0.0, 0.0, 0.5, 1.0, 0.55, 0.1, 0.1],
        atol=1e-6,
    )
    constant = lr_schedule(2.0)
    assert float(constant(0)) == float(constant(10**6)) == 2.0


def test_separate_lr_applies_the_schedule_to_updates():
    """With a constant gradient, Adam's update size equals the scheduled learning rate."""
    from brax.training.agents.ppo.losses import PPONetworkParams

    from lsy_drone_racing.control.rl_race.optim import separate_lr

    params = PPONetworkParams(policy={"w": jnp.zeros(2)}, value={"w": jnp.zeros(2)})
    grads = PPONetworkParams(policy={"w": jnp.ones(2)}, value={"w": jnp.ones(2)})
    opt = separate_lr(
        1e-4, 3e-4, max_grad_norm=None, warmup_steps=10, decay_steps=100, final_lr_factor=0.1
    )
    state = opt.init(params)
    sizes = []
    for _ in range(130):
        upd, state = opt.update(grads, state, params)
        sizes.append((float(-upd.policy["w"][0]), float(-upd.value["w"][0])))
    for step, (pol, val) in {
        5: (0.5e-4, 1.5e-4),
        10: (1e-4, 3e-4),
        60: (0.55e-4, 1.65e-4),
        129: (1e-5, 3e-5),
    }.items():
        np.testing.assert_allclose(sizes[step], (pol, val), rtol=1e-3)


def test_four_action_independent_limits_and_history():
    cfg = ActionConfig(
        roll_limit_deg=20, pitch_limit_deg=40, yaw_limit_deg=70, thrust_min=0.1, thrust_max=0.9
    )
    command = to_command(jnp.array([[2.0, -2.0, 0.5, 0.0]]), cfg)
    np.testing.assert_allclose(
        command, [[np.deg2rad(20), -np.deg2rad(40), np.deg2rad(35), 0.5]], atol=1e-6
    )
    previous = jnp.array([[0.1, 0.2, -0.3, 0.4]])
    features = encode(_raw([0.0, 0.0, 1.0]), previous, ObsConfig())
    assert features.shape == (1, 49)
    np.testing.assert_allclose(features[0, 18:22], previous[0])
    with pytest.raises(ValueError, match="four actions"):
        to_command(jnp.zeros((1, 3)), cfg)


def test_thrust_fraction_config_and_physical_bounds():
    from lsy_drone_racing.control.rl_race.train import env_config

    cfg = env_config("race", 70, 60, 50, 0.5, 750, thrust_min_fraction=0.2, thrust_max_fraction=0.8)
    assert cfg.action.thrust_min_fraction == 0.2
    assert cfg.action.thrust_max_fraction == 0.8
    for low, high in [(-0.1, 1), (0, 1.1), (0.8, 0.2), (0.5, 0.5)]:
        with pytest.raises(ValueError, match="Thrust fractions"):
            ActionConfig(thrust_min_fraction=low, thrust_max_fraction=high)


def test_network_config_factory_preserves_brax_networks():
    import jax
    from lsy_drone_racing.control.rl_race.network import NetworkConfig, make_networks

    cfg = NetworkConfig((32, 32), (32,), "swish")
    actual = cfg.factory()(49, 4)
    expected = make_networks(
        49,
        4,
        policy_hidden_layer_sizes=(32, 32),
        value_hidden_layer_sizes=(32,),
        activation=jax.nn.swish,
        distribution_type="tanh_normal",
    )
    key = jax.random.PRNGKey(17)
    for a, b in zip(
        jax.tree.leaves(actual.policy_network.init(key)),
        jax.tree.leaves(expected.policy_network.init(key)),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="positive"):
        NetworkConfig((0,), (32,))
    with pytest.raises(ValueError, match="noise_scale"):
        NetworkConfig(noise_scale=0)
