"""Competition evaluation script.

Note:
    Please do not alter this script or ask the course supervisors first!
"""

import logging
import sys
from pathlib import Path

import numpy as np
from sim import simulate

from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)


def main():
    """Run the simulation N times and save the results as 'evaluation.csv'."""
    n_runs = 20
    config_file = "level2.toml"
    config = load_config(Path(__file__).parents[1] / "config" / config_file)
    ep_times = simulate(
        config=config_file, controller=config.controller.file, n_runs=n_runs, render=False
    )

    # Log the number of failed runs if any
    if n_failed := len([x for x in ep_times if x is None]):
        logger.warning(f"{n_failed} run{'' if n_failed == 1 else 's'} failed out of {n_runs}!")
    else:
        logger.info("All runs completed successfully!")

    # Abort if more than half of the runs failed
    if (success_rate := 1 - n_failed / n_runs) < 0.5:
        print(f"::error title=Evaluation Failed::{n_failed} of {n_runs} runs failed (more than 50%)!")
        sys.exit(1)

    successful_times_avg = np.mean([x for x in ep_times if x is not None])
    logger.info(f"Average Time: successful_times_avg (s): {successful_times_avg}")
    logger.info(f"Success Rate: {success_rate * 100}%")
    file = Path(__file__).parents[1] / "evaluation.csv"
    with open(file, "w") as f:
        f.write(f"{successful_times_avg},{success_rate},")
    logger.info(f"Results saved in {file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
