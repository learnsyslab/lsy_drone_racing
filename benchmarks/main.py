from __future__ import annotations

import fire
import numpy as np
from sim import time_multi_drone_reset, time_multi_drone_step, time_sim_reset, time_sim_step


def print_benchmark_results(name: str, timings: list[float], n_envs: int, device: str):
    print(f"\nResults for {name} ({n_envs} envs, {device}):")
    print(f"Mean/std: {np.mean(timings):.2e}s +- {np.std(timings):.2e}s")
    print(f"Min time: {np.min(timings):.2e}s")
    print(f"Max time: {np.max(timings):.2e}s")
    print(f"FPS: {n_envs / np.mean(timings):.2f}")


def main(
    n_tests: int = 2,
    number: int = 100,
    multi_drone: bool = False,
    reset: bool = True,
    step: bool = True,
    vec_size: int = 1,
    device: str = "cpu",
):
    reset_fn, step_fn = time_sim_reset, time_sim_step
    if multi_drone:
        reset_fn, step_fn = time_multi_drone_reset, time_multi_drone_step
    if reset:
        timings = reset_fn(n_tests=n_tests, number=number, n_envs=vec_size, device=device)
        print_benchmark_results(
            name="Racing env reset", timings=timings / number, n_envs=vec_size, device=device
        )
    if step:
        timings = step_fn(n_tests=n_tests, number=number, n_envs=vec_size, device=device)
        print_benchmark_results(
            name="Racing env steps", timings=timings / number, n_envs=vec_size, device=device
        )
        timings = step_fn(
            n_tests=n_tests, number=number, dynamics="so_rpy", n_envs=vec_size, device=device
        )
        print_benchmark_results(
            name="Racing env steps (so_rpy backend)",
            timings=timings / number,
            n_envs=vec_size,
            device=device,
        )


if __name__ == "__main__":
    fire.Fire(main)
