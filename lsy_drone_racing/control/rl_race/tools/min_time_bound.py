"""Point-mass lower bound on the lap time of the nominal level-2 track.

The drone is a point mass with a thrust vector f limited by
    f_min <= |f| <= f_max (4 x the per-rotor limits)  and  tilt(f) <= theta_max,
plus gravity. Rotational dynamics, the onboard attitude controller, drag and disturbances are
ignored, so the true minimum lap time of the quadrotor is never shorter than this bound (up to the
gate and obstacle margins below and the mass randomization, see --mass).

The lap is split into one phase per gate pass (start -> gate 1, ..., gate 4 -> gate 2), each with
N nodes and its own time step. Each phase ends in the gate's plane, inside the opening shrunk by
the drone radius, while moving forward through it. Obstacles are vertical cylinders (checked at
nodes and interval midpoints). The lap ends when the last gate is passed, as in the env.

Usage:
    python -m lsy_drone_racing.control.rl_race.tools.min_time_bound --theta_max_deg=85
"""

from __future__ import annotations

import json
from pathlib import Path

import casadi as ca
import fire
import numpy as np

from lsy_drone_racing.utils import load_config

ROOT = Path(__file__).resolve().parents[4]
G = 9.81


def track(config: str = "level2.toml") -> dict:
    """Nominal gates, obstacles, start and drone limits of a level config."""
    from crazyflow.dynamics import load_params

    cfg = load_config(ROOT / "config" / config)
    t = cfg.env.track
    gates = np.array([g["pos"] for g in t.gates], float)
    yaws = np.array([g["rpy"][2] for g in t.gates], float)
    order = [int(o) for o in cfg.env.track.get("gate_order", [1, 2, 3, 4])]
    params = load_params(cfg.sim.dynamics, cfg.sim.drone)
    return {
        "gates": gates,
        "normals": np.stack([np.cos(yaws), np.sin(yaws), np.zeros_like(yaws)], 1),
        "order": [abs(o) - 1 for o in order],
        "sign": [1.0 if o > 0 else -1.0 for o in order],
        "obstacles": np.array([o["pos"] for o in t.obstacles], float),
        "start": np.array(t.drones[0]["pos"], float),
        "mass": float(params["mass"]),
        "f_max": 4 * float(params["thrust_max"]),
        "f_min": 4 * float(params["thrust_min"]),
        "low": np.array(t.safety_limits["pos_limit_low"], float),
        "high": np.array(t.safety_limits["pos_limit_high"], float),
    }


def solve(
    theta_max_deg: float = 85.0,
    margin: float = 0.05,
    opening: float = 0.4,
    obstacle_radius: float = 0.015,
    nodes: int = 30,
    mass: float | None = None,
    guess_speed: float = 3.0,
    rate_max: float | None = 8.0,
    acc_max: float = 160.0,
    config: str = "level2.toml",
    out: str | None = None,
    verbose: bool = False,
) -> dict:
    """Solve the minimum-time problem and return the lap time and per-leg times.

    Args:
        theta_max_deg: Maximum tilt of the thrust vector.
        margin: Drone radius: the gate opening shrinks and obstacles grow by this much.
        opening: Side length of the square gate opening (m).
        obstacle_radius: Obstacle cylinder radius (m).
        nodes: Nodes per leg.
        mass: Override the nominal mass (kg); level 2 randomizes it by +-5 g.
        guess_speed: Speed of the straight-line initial guess (m/s).
        rate_max: Turn-rate limit of the thrust direction (rad/s); None = pure point mass (the
            thrust direction may jump). Measured in the simulator: ~8 rad/s
            (outputs/debug/attitude_step.py).
        acc_max: Angular-acceleration limit of the thrust direction (rad/s^2); measured 120-160.
        config: Level config in config/.
        out: Optional path for a JSON dump of the trajectory.
        verbose: Print IPOPT output.
    """
    tr = track(config)
    m = mass or tr["mass"]
    half = opening / 2 - margin
    legs = len(tr["order"])
    cos_t = np.cos(np.deg2rad(theta_max_deg))

    def build(relaxed: bool, attitude: float | None, warm: list | None = None) -> tuple:
        """Build one stage of the constrained trajectory problem.

        Relaxed mode removes min thrust and obstacles; attitude limits turning rate.
        """
        opti = ca.Opti()
        dts = opti.variable(legs)
        P, V, F, B, W, U, T = [], [], [], [], [], [], []
        for _ in range(legs):
            P.append(opti.variable(3, nodes + 1))
            V.append(opti.variable(3, nodes + 1))
            F.append(opti.variable(3, nodes))
            if attitude:
                B.append(opti.variable(3, nodes + 1))  # thrust direction (unit)
                W.append(opti.variable(3, nodes + 1))  # its angular velocity
                U.append(opti.variable(3, nodes))  # angular acceleration
                T.append(opti.variable(1, nodes))  # thrust magnitude
        gvec = ca.DM([0, 0, -G])
        opti.subject_to(P[0][:, 0] == tr["start"])
        opti.subject_to(V[0][:, 0] == 0)
        if attitude:
            opti.subject_to(B[0][:, 0] == ca.DM([0, 0, 1]))
            opti.subject_to(W[0][:, 0] == 0)
        for k in range(legs):
            p, v, f, dt = P[k], V[k], F[k], dts[k]
            opti.subject_to(opti.bounded(0.005, dt, 0.2))
            if k > 0:  # continuity between legs
                opti.subject_to(p[:, 0] == P[k - 1][:, -1])
                opti.subject_to(v[:, 0] == V[k - 1][:, -1])
                if attitude:
                    opti.subject_to(B[k][:, 0] == B[k - 1][:, -1])
                    opti.subject_to(W[k][:, 0] == W[k - 1][:, -1])
            for i in range(nodes):
                if attitude:  # f along b, b turns with w, |w| and |dw/dt| bounded
                    b, w = B[k][:, i], W[k][:, i]
                    opti.subject_to(f[:, i] == T[k][i] * b)
                    nb = b + dt * ca.cross(w, b)
                    opti.subject_to(B[k][:, i + 1] == nb / ca.sqrt(ca.sumsqr(nb)))
                    opti.subject_to(W[k][:, i + 1] == w + dt * U[k][:, i])
                    opti.subject_to(ca.sumsqr(W[k][:, i + 1]) <= attitude**2)
                    opti.subject_to(ca.sumsqr(U[k][:, i]) <= acc_max**2)
                a = f[:, i] / m + gvec
                opti.subject_to(p[:, i + 1] == p[:, i] + v[:, i] * dt + 0.5 * a * dt**2)
                opti.subject_to(v[:, i + 1] == v[:, i] + a * dt)
                fn2 = ca.sumsqr(f[:, i])
                opti.subject_to(fn2 <= tr["f_max"] ** 2)
                if not relaxed:
                    opti.subject_to(fn2 >= tr["f_min"] ** 2)
                opti.subject_to(f[2, i] >= cos_t * ca.sqrt(fn2 + 1e-9))
                mid = p[:, i] + 0.5 * v[:, i] * dt + 0.125 * a * dt**2
                for q in () if relaxed else (p[:, i + 1], mid):
                    for o in tr["obstacles"]:
                        opti.subject_to(
                            (q[0] - o[0]) ** 2 + (q[1] - o[1]) ** 2
                            >= (obstacle_radius + margin) ** 2
                        )
            for i in range(1, nodes + 1):  # stay in the arena and above the start height
                opti.subject_to(opti.bounded(tr["low"][:2], p[:2, i], tr["high"][:2]))
                opti.subject_to(opti.bounded(tr["start"][2], p[2, i], tr["high"][2]))
            # Leg ends in the gate plane, inside the shrunk opening, moving forward through it.
            g = tr["order"][k]
            c, n = tr["gates"][g], tr["normals"][g] * tr["sign"][k]
            lat = np.array([-n[1], n[0], 0.0])
            d = p[:, -1] - c
            opti.subject_to(ca.dot(d, n) == 0)
            opti.subject_to(opti.bounded(-half, ca.dot(d, lat), half))
            opti.subject_to(opti.bounded(-half, d[2], half))
            opti.subject_to(ca.dot(v[:, -1], n) >= 0.5)
        opti.minimize(nodes * ca.sum1(dts))

        if warm is not None:  # previous solution
            for var, val in zip([dts, *P, *V, *F], warm):
                opti.set_initial(var, val)
            if attitude and len(warm) > 1 + 3 * legs:  # previous attitude stage
                for var, val in zip([*B, *W, *U, *T], warm[1 + 3 * legs :]):
                    opti.set_initial(var, val)
            elif attitude:  # thrust direction and its rates from the point-mass thrust
                dtv = np.asarray(warm[0]).ravel()
                for k in range(legs):
                    fk = np.asarray(warm[1 + 2 * legs + k]).reshape(3, nodes)
                    bk = fk / np.maximum(np.linalg.norm(fk, axis=0), 1e-6)
                    bk = np.concatenate([bk, bk[:, -1:]], axis=1)
                    wk = np.cross(bk[:, :-1].T, np.diff(bk, axis=1).T).T / dtv[k]
                    wk = np.concatenate([wk, wk[:, -1:]], axis=1)
                    wk *= np.minimum(1.0, attitude / np.maximum(np.linalg.norm(wk, axis=0), 1e-6))
                    opti.set_initial(B[k], bk)
                    opti.set_initial(W[k], wk)
                    opti.set_initial(U[k], np.diff(wk, axis=1) / dtv[k])
                    opti.set_initial(T[k], np.linalg.norm(fk, axis=0)[None])
            return opti, dts, P, V, F, B + W + U + T
        # Initial guess: straight lines through the gate centers at guess_speed, hover thrust.
        pts = [tr["start"]] + [tr["gates"][g] for g in tr["order"]]
        for k in range(legs):
            a, b = pts[k], pts[k + 1]
            dist = np.linalg.norm(b - a)
            opti.set_initial(dts[k], dist / guess_speed / nodes)
            s = np.linspace(0, 1, nodes + 1)
            opti.set_initial(P[k], (a[:, None] + (b - a)[:, None] * s[None]))
            opti.set_initial(V[k], np.tile(((b - a) / dist * guess_speed)[:, None], nodes + 1))
            opti.set_initial(F[k], np.tile(np.array([[0], [0], [m * G]]), nodes))
        return opti, dts, P, V, F, B + W + U + T

    dts, P, V, F = None, [], [], []
    sol = None
    # Homotopy: relaxed point mass -> point mass -> rate limit tightened step by step.
    stages = [(True, None), (False, None)]
    if rate_max:
        stages += [(False, r) for r in (30.0, 15.0, 12.0, 10.0, 9.0, 8.5) if r > rate_max] + [
            (False, rate_max)
        ]
    extra: list = []
    for relaxed, attitude in stages:
        warm = (
            None if sol is None else [np.asarray(sol.value(x)) for x in [dts, *P, *V, *F, *extra]]
        )
        opti, dts, P, V, F, extra = build(relaxed, attitude, warm)
        sol, ok = _solve(opti, verbose)
        if verbose or not ok:
            stats = opti.stats()
            print(
                f"stage relaxed={relaxed} rate={attitude}: {stats['return_status']}, "
                f"lap {nodes * float(np.sum(sol.value(dts))):.3f} s"
            )
        if not ok and not relaxed:  # the relaxed stage only provides a warm start
            break
    dt = np.asarray(sol.value(dts)).ravel()
    leg_t = dt * nodes
    p = np.concatenate(
        [np.asarray(sol.value(P[k]))[:, :-1] for k in range(legs)]
        + [np.asarray(sol.value(P[-1]))[:, -1:]],
        axis=1,
    )
    v = np.concatenate(
        [np.asarray(sol.value(V[k]))[:, :-1] for k in range(legs)]
        + [np.asarray(sol.value(V[-1]))[:, -1:]],
        axis=1,
    )
    f = np.concatenate([np.asarray(sol.value(F[k])) for k in range(legs)], axis=1)
    tilt = np.degrees(np.arccos(np.clip(f[2] / np.linalg.norm(f, axis=0), -1, 1)))
    result = {
        "converged": ok,
        "lap_time": float(leg_t.sum()),
        "leg_times": leg_t.round(3).tolist(),
        "max_speed": float(np.linalg.norm(v, axis=0).max()),
        "tilt_at_limit": float(np.mean(tilt > theta_max_deg - 1)),
        "thrust_at_max": float(np.mean(np.linalg.norm(f, axis=0) > 0.99 * tr["f_max"])),
        "settings": {
            "theta_max_deg": theta_max_deg,
            "margin": margin,
            "mass": m,
            "rate_max": rate_max,
            "acc_max": acc_max if rate_max else None,
            "a_max_g": tr["f_max"] / m / G,
        },
    }
    if out:
        t = np.concatenate([[0.0], np.cumsum(np.repeat(dt, nodes))])
        Path(out).write_text(
            json.dumps(
                {
                    **result,
                    "t": t.tolist(),
                    "pos": p.T.tolist(),
                    "vel": v.T.tolist(),
                    "thrust": f.T.tolist(),
                }
            )
        )
    return result


def _solve(opti: ca.Opti, verbose: bool) -> tuple:
    """Run IPOPT; return (solution or debug values, converged)."""
    opti.solver(
        "ipopt",
        {"print_time": False},
        {"max_iter": 5000, "print_level": 5 if verbose else 0, "tol": 1e-6},
    )
    try:
        return opti.solve(), True
    except RuntimeError:
        return opti.debug, False


def main(sweep: bool = False, **kwargs: object) -> None:
    """Solve once (kwargs as in ``solve``), or a small sweep over tilt, margin, mass and guesses."""
    if not sweep:
        print(json.dumps(solve(**kwargs), indent=2))
        return
    tr = track()
    for theta in (85.0, 90.0):
        for margin in (0.0, 0.05):
            for mass in (tr["mass"], tr["mass"] - 0.005):
                best = None
                for speed in (2.0, 3.0, 4.0):
                    r = solve(theta, margin, mass=mass, guess_speed=speed)
                    if r["converged"] and (best is None or r["lap_time"] < best["lap_time"]):
                        best = r
                if best is None:
                    print(
                        f"theta {theta:4.0f} margin {margin:.2f} "
                        f"mass {mass * 1e3:4.1f} g: no solution"
                    )
                    continue
                print(
                    f"theta {theta:4.0f} margin {margin:.2f} mass {mass * 1e3:4.1f} g: "
                    f"lap {best['lap_time']:.3f} s, legs {best['leg_times']}, "
                    f"v_max {best['max_speed']:.1f} m/s, thrust at max {best['thrust_at_max']:.0%}"
                )


if __name__ == "__main__":
    fire.Fire(main)
