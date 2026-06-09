# Autonomous Drone Racing Challenge

<div align="center">
  <a href="https://github.com/learnsyslab/crazyflow">
    <img width="800" height="450" src="docs/img/race.gif">
  </a>
  <br/>
  <sub><sup style="font-size: 0.8em;"><a href="https://github.com/learnsyslab/crazyflow">Powered by Crazyflow</a></sup></sub>
  </br>
  </br>
</div>

[![Python Version]][Python Version URL] [![Ruff Check]][Ruff Check URL] [![Documentation Status]][Documentation Status URL] [![Tests]][Tests URL]

[Python Version]: https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue  
[Python Version URL]: https://www.python.org  

[Ruff Check]: https://github.com/learnsyslab/lsy_drone_racing/actions/workflows/ruff.yml/badge.svg?style=flat-square  
[Ruff Check URL]: https://github.com/learnsyslab/lsy_drone_racing/actions/workflows/ruff.yml  

[Documentation Status]: https://readthedocs.org/projects/lsy-drone-racing/badge/?version=latest  
[Documentation Status URL]: https://lsy-drone-racing.readthedocs.io/en/latest/?badge=latest  

[Tests]: https://github.com/learnsyslab/lsy_drone_racing/actions/workflows/testing.yml/badge.svg  
[Tests URL]: https://github.com/learnsyslab/lsy_drone_racing/actions/workflows/testing.yml  

---

## Introduction

**LSY Drone Racing** is a course project designed to help you develop and evaluate autonomous drone racing algorithms — both in simulation and on real Crazyflie hardware.  
Whether you’re new to drones or an experienced developer, this project provides a structured and practical way to explore high-speed autonomy, control, and perception in dynamic environments.

---

## Documentation

To get started, visit our [official documentation](https://lsy-drone-racing.readthedocs.io/en/latest/getting_started/general.html).

---

## Dependencies

This project builds upon several open-source packages developed by the [Learning Systems Lab (LSY)](https://www.ce.cit.tum.de/lsy/home/) at TUM.  
You can explore these related projects:

- [**crazyflow**](https://github.com/learnsyslab/crazyflow) – A high-speed, high-fidelity drone simulator with strong sim-to-real performance.  
- [**drone-models**](https://github.com/learnsyslab/drone-models) – A collection of accurate drone models for simulation and model-based control.  
- [**drone-controllers**](https://github.com/learnsyslab/drone-controllers) – Controllers for the Crazyflie quadrotor.  

---

## Difficulty Levels

Each task setup — from track design to physics configuration — is defined by a TOML file (e.g., [`level0.toml`](config/level0.toml)).  
The configuration files specify progressive difficulty levels from easy (0) to hard (3):

|      Evaluation Scenario      | Rand. Inertial Properties | Randomized Obstacles, Gates | Random Tracks |             Notes              |
| :---------------------------: | :-----------------------: | :-------------------------: | :-----------: | :----------------------------: |
| [Level 0](config/level0.toml) |           *No*            |            *No*             |     *No*      |       Perfect knowledge        |
| [Level 1](config/level1.toml) |          **Yes**          |            *No*             |     *No*      |        Adaptive control        |
| [Level 2](config/level2.toml) |          **Yes**          |           **Yes**           |     *No*      |          Re-planning           |
| [Level 3](config/level3.toml) |          **Yes**          |           **Yes**           |    **Yes**    |        Online planning         |
|         **sim2real**          |     **Real hardware**     |           **Yes**           |    **Yes**    | Simulation-to-reality transfer |

---

## Online Competition

Throughout the semester, teams in the course will compete to achieve the fastest autonomous race completion times. 
Competition results visible in the [competition branch](https://github.com/learnsyslab/lsy_drone_racing/tree/competition).

> **Note:** Competition results **do not** directly affect your course grade.  
> However, they provide valuable feedback on the performance and robustness of your approach compared to others.

The competition environment always uses **difficulty level 2**.  
If your code fails the automated tests, it is likely to encounter the same issues in our evaluation environment.  
For full details, refer to the [documentation](https://lsy-drone-racing.readthedocs.io/en/latest/).

---
