"""This module contains the environments for the drone racing challenge.

Environments are split into simulation and real-world environments. The simulation environments use
the simulation module to provide a realistic simulation of the drone racing challenge for training,
testing and iterating on controller designs. The real-world environments mirror the interface of the
simulation environments, but use the Vicon motion capture system to track the drone and race track
elements in the lab, and sent the controller actions to the real drone.

Note:
    While the interfaces are the same and we try to keep the environments as similar as possible,
    the dynamics of the drone and all observations are subject to a sim2real gap. The transition
    between simulation and real-world may therefore require additional tuning of the controller
    design.
"""

from gymnasium import register

# region SimEnvs

register(
    id="DroneRacing-v0",
    entry_point="lsy_drone_racing.envs.drone_race:DroneRaceEnv",
    vector_entry_point="lsy_drone_racing.envs.drone_race:VecDroneRaceEnv",
    max_episode_steps=1500,  # 30 seconds * 50 Hz,
    disable_env_checker=True,  # Remove warnings about 2D observations
)

register(
    id="MultiDroneRacing-v0",
    entry_point="lsy_drone_racing.envs.multi_drone_race:MultiDroneRaceEnv",
    vector_entry_point="lsy_drone_racing.envs.multi_drone_race:VecMultiDroneRaceEnv",
    max_episode_steps=1500,
    disable_env_checker=True,
)

# region RealEnvs

register(
    id="RealDroneRacing-v0",
    entry_point="lsy_drone_racing.envs.real_race_env:RealDroneRaceEnv",
    disable_env_checker=True,
)

register(
    id="RealMultiDroneRacing-v0",
    entry_point="lsy_drone_racing.envs.real_race_env:RealMultiDroneRaceEnv",
    disable_env_checker=True,
)

register(
    id="RealMultiDroneRaceEnvClient-v0",
    entry_point="lsy_drone_racing.envs.real_race_client_env:RealMultiDroneRaceEnvClient",
    disable_env_checker=True,
)
