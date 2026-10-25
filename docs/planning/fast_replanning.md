# Fast re-planning under 0.7 m sensing

This note answers one question: *which path-planning algorithms can re-plan fast enough for a
quadrotor that only learns the true pose of gates and poles within 0.7 m of them?* It starts from a
measurement of the planner that is currently in the repository, because that measurement changes
the answer.

## 1. The constraint, in numbers

Taken from `config/level2.toml` and `MPCCConfig`:

| Quantity | Value |
| --- | --- |
| Track | 4 gates, 5 ordered passes, 4 pole obstacles, arena 5 x 3 x 2 m |
| `sensor_range` | 0.7 m |
| Gate pose error (uniform) | +-0.15 m x/y, +-0.10 m z, +-0.2 rad yaw |
| Obstacle pose error (uniform) | +-0.15 m x/y, +-0.05 m z |
| Worst-case lateral gate error | `sqrt(0.15^2 + 0.15^2)` = 0.21 m |
| Gate opening | 0.40 m wide (half opening 0.20 m) |
| `v_max` | 2.3 m/s |
| Environment step | 50 Hz, 20 ms per control cycle |
| Re-plan per reveal | 1 control cycle, so the budget is 20 ms |

Two consequences drive the design:

1. **The correction is mandatory.** The worst-case gate offset (0.21 m) is larger than the gate half
   opening (0.20 m), so flying the nominal pose is not a safe fallback. The reference has to be
   corrected on reveal.
2. **The reaction time is short but not zero.** 0.7 m at 2.3 m/s is 0.30 s, i.e. about 15 control
   cycles. There is enough time for a *small* correction, and not enough time for a large one, which
   is exactly the regime where incremental local repair beats global re-search.

## 2. What the current planner costs

`PathPlanner.plan` on `level2.toml` poses, 30 randomised draws, pixi environment, this machine:

| Configuration | mean | p95 | max |
| --- | --- | --- | --- |
| Default `PathConfig` | 60 ms | 77 ms | 84 ms |
| `smooth_iterations = 60` | 14 ms | 20 ms | 20 ms |
| `smooth_iterations = 0` | 9 ms | 12 ms | 15 ms |

`cProfile` attributes **86 % of the total to `_smooth_path`**: it runs 600 Laplacian sweeps over a
polyline of ~1500 points and, inside every sweep, a full `np.linalg.norm` over a
`(1500, 4, 2)` difference array plus an `argmin`.

So the current re-plan is **3x to 5x over the entire control period**. Each reveal stalls the 50 Hz
loop for three to five cycles. With up to 8 reveals per lap (4 gates + 4 poles) that is 0.5-0.8 s of
lost control per lap - more than the total latency of any planning algorithm discussed below, and
larger than the time the algorithm choice can buy back.

A closed-form construction of the same path measures the achievable floor:

| Stage | mean | p95 |
| --- | --- | --- |
| `ArcPath` fit through ~18 waypoints + 400 samples | 0.14 ms | 0.18 ms |
| Vectorised push-out, 400 points x 4 poles, all poles at once | 0.06 ms | 0.09 ms |
| Relaxation on 60 points instead of 1500, 40 sweeps | 0.12 ms | 0.21 ms |
| **Sum (a complete re-plan)** | **~0.3 ms** | **~0.5 ms** |

The floor is ~0.5 ms, i.e. **1 % of the control period and two orders of magnitude below the current
planner**. The lesson: in this repository the algorithm choice is not the bottleneck, the
implementation of the smoothing loop is.

## 3. Why "path finding" is the wrong abstraction here

A grid, sampling or lattice search buys you one thing: finding a route when you do not know which
route exists. Neither precondition holds in this race:

* **The order is fixed by the rules.** The gate sequence and direction are given, so there is no
  global route decision to search for. The path is a chain of five constrained passes, and every pass
  is a boundary-value problem between two known planes, not a search problem.
* **The only discrete decisions are tiny.** The obstacle set is four thin poles, so the homotopy
  choices are "which side of each pole", at most `2^4 = 16` combinations. Enumerating all 16 and
  scoring them costs microseconds; a search tree that discovers them costs milliseconds.
* **Free space is convex-ish.** A 5 x 3 x 2 m arena with four 0.03 m cylinders has no narrow
  corridors, so there is no need for the completeness guarantee that makes grid search attractive.

A* on a 3D voxel grid fine enough to resolve 0.03 m poles and 0.09 m drone clearance over this arena
needs on the order of `10^5`-`10^6` nodes, i.e. tens to hundreds of milliseconds in Python. RRT* and
BIT* land in the same class and add run-to-run nondeterminism, which is exactly what you do not want
under a hard deadline. They are the right answer to a different problem (unknown, cluttered,
non-convex environments with no prescribed order) and the wrong answer to this one.

## 4. The candidate algorithms, sorted by cost class

Latency figures are order-of-magnitude, as reported in the cited work or measured here; they are
listed to rank the families, not as benchmark claims on this hardware.

| Family | Typical re-plan | Deterministic | Dynamically feasible output | Fit for this track |
| --- | --- | --- | --- | --- |
| **Closed-form spline through ordered waypoints** (what this repo does, minus the relaxation) | 0.1-1 ms (measured here) | yes | C2, curvature-limitable | **Best.** Order is fixed; only pole side choice and gate centre correction remain |
| **Warm-started B-spline / MINCO trajectory optimisation** ([MINCO, T-RO 2022](https://arxiv.org/abs/2103.00190)) | 0.5-5 ms | yes | yes, time-optimal-ish | **Best generic upgrade.** Same cost class, gives clearance and time allocation for free |
| **ESDF-free gradient repair** ([EGO-Planner, RA-L 2021](https://arxiv.org/abs/2008.08835)) | 1-10 ms | yes | yes | Very good for the *local* repair layer on reveal |
| **Convex corridor + QP** ([Safe Flight Corridors, RA-L 2017](https://ieeexplore.ieee.org/abstract/document/7839930)) | 2-20 ms | yes | yes | Good clearance guarantee, heavier than needed for 4 poles |
| **Kinodynamic / lattice A\*** ([Fast-Planner, RA-L 2019](https://arxiv.org/abs/1907.01531)) | 5-50 ms | yes | yes | Only if the gate order were free or the arena became cluttered |
| **Visibility graph / bitangent graph** | 0.01-0.5 ms for 4 poles | yes | no (needs smoothing) | Excellent for the *discrete* pole-side choice, not for the smooth path |
| **RRT / RRT* / BIT\*** | 10-500 ms | no | no | Too slow, too variable |
| **3D A\* on a fine voxel grid** | 50-1000 ms | yes | no | Too slow at this resolution |
| **Fast marching / ESDF over a grid** | 20-500 ms | yes | no | Same problem, and the field update dominates |
| **Learned policy / RL end-to-end** | 0.1-1 ms | yes | learned | Interesting, but needs training and does not use the known gate order |

The three names worth reading in full are EGO-Planner (repair without an ESDF, "only extract obstacle
information when the trajectory actually hits something"), MINCO (a trajectory representation whose
spatial-temporal deformation is linear-time, which is what makes 1 ms re-planning possible), and
Fast-Planner (kinodynamic search plus B-spline refinement, for the case where you *do* need
re-search).

## 5. Recommended architecture

Three layers, each with a hard latency ceiling. Only the first two exist today.

### L0 - once per lap: ordered-gate construction
Build the waypoint chain from the gate centres, approach/exit offsets and pole detours, fit the arc
length spline, done. This is today's `plan()`, and it is the right *structure*; it only needs to stop
costing 60 ms. Push the smoothing onto the sparse waypoint polyline (tens of points, not 1500) or
replace it with a G2 corner blend, and vectorise the push-out over all poles at once. Target: **< 1 ms**
(measured floor above).

### L1 - on reveal: monotone local repair, never a full re-plan
On a pose change, repair only the arc-length window that the reveal can affect:

```
s = path.closest(drone_pos)                  # arc length of the drone
s0 = s - 0.3                                 # a little behind, for continuity
s1 = s + v_max * 1.5                         # ~1.5 s of lookahead at most
# 1. drop waypoints of the affected gates/poles inside [s0, s1]
# 2. re-place them at the revealed poses (gate centre, pole side)
# 3. re-fit only that segment, clamped at both ends to position + tangent
# 4. keep everything beyond s1 bit-identical, reparameterise locally
```

Two properties make this safe and fast:

* **Warm start.** The previous path is the initial guess, and a reveal only moves a waypoint by at
  most 0.21 m, so one or two Gauss-Newton / relaxation passes converge. This is the EGO-Planner
  argument: only the part of the trajectory that actually collides needs attention.
* **Monotone tail.** The tail beyond the repair window is unchanged, so `theta` and the MPCC stage
  references stay consistent and the solver does not have to re-converge from scratch.

Target: **< 1 ms**, and in the common case (a pole revealed far from the current segment) **0**.

### L2 - in the MPC: make pole avoidance cost no re-plan at all
The OCP currently carries `NP = 7` parameters per stage: reference position (3), tangent (3), and the
contouring weight (1). Poles influence the solution only indirectly, through the path shape.

Add the *nearest pole* to the stage parameters - `(x, y, r)` - giving `NP = 10`, and add a soft
clearance penalty on the distance from the predicted position to that pole. A reveal then becomes a
parameter write (`N + 1` stages x 10 doubles, tens of microseconds) instead of a re-plan: the solver
that is already running at 50 Hz absorbs the new obstacle in the next SQP-RTI iteration. This is the
standard way to get obstacle avoidance for free in an MPC, and it removes the pole half of the reveal
events entirely.

Two practical traps when changing `NP`:

* `_signature()` in `my_controller.py` hashes only the config, the mass, the thrust limits, the lag
  coefficients and gravity. It does **not** hash `NP`, so a changed `NP` would silently reuse the
  stale `c_generated_code` solver. Add `NP`, `NX`, `NU` to the payload, or clear the directory.
* The avoidance term must be a *soft* penalty (or a linearised half-plane constraint along the
  current obstacle-to-drone direction) because the exact clearance constraint is non-convex and would
  fight `SQP_RTI` at 50 Hz.

## 6. Change list, in order of payoff

1. **Sparse-polyline smoothing + vectorised push-out.** ~60 ms -> ~0.5 ms, no change to the plan
   itself. Biggest single win; nothing else matters until this is done.
2. **Local repair window in `_replan`.** Stop re-planning the entire remaining track when a reveal
   only affects the next 1.5 m. Makes the cost proportional to what changed.
3. **Pole avoidance as an OCP parameter (`NP = 7 -> 10`).** Removes the pole reveals from the re-plan
   path entirely, and improves clearance because the solver reacts every cycle instead of only at the
   reveal event.
4. **Cache the plan per reveal event, not per step.** `_track_info_changed` already compares poses;
   also require that the changed object is within some distance of the affected segment, so a reveal
   behind the drone cannot trigger a re-plan.

If steps 1-2 are done, the re-plan fits in the 20 ms budget with two orders of magnitude of margin,
and a full global re-search is never needed for this track. Step 3 is worth doing anyway, because it
removes a whole class of deadline risk.

## 7. References

* X. Zhou, Z. Wang, H. Ye, C. Xu, F. Gao, *EGO-Planner: An ESDF-free Gradient-based Local Planner for
  Quadrotors*, RA-L 2021. [arXiv:2008.08835](https://arxiv.org/abs/2008.08835)
* Z. Wang, X. Zhou, C. Xu, F. Gao, *Geometrically Constrained Trajectory Optimization for Multicopters*
  (MINCO / GCopter), T-RO 2022. [arXiv:2103.00190](https://arxiv.org/abs/2103.00190)
* B. Zhou, F. Gao, L. Wang, C. Liu, S. Shen, *Robust and Efficient Quadrotor Trajectory Generation for
  Fast Autonomous Flight* (Fast-Planner), RA-L 2019. [arXiv:1907.01531](https://arxiv.org/abs/1907.01531)
* S. Liu, M. Watterson, K. Mohta, K. Sun, S. Bhattacharya, C. J. Taylor, V. Kumar, *Planning
  Dynamically Feasible Trajectories for Quadrotors Using Safe Flight Corridors in 3-D Complex
  Environments*, RA-L 2017.
  [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/7839930)
* A. Romero, S. Sun, P. Foehn, D. Scaramuzza, *Model Predictive Contouring Control for Time-Optimal
  Quadrotor Flight*, T-RO 2022. [PDF](https://rpg.ifi.uzh.ch/docs/TRO22_Romero.pdf) - the formulation
  already implemented in `my_controller.py`.
* S. Karaman, E. Frazzoli, *Sampling-based Algorithms for Optimal Motion Planning*, IJRR 2011 (the
  RRT* reference; cited here for why the sampling family is the wrong latency class for this track).
