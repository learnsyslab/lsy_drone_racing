Overview
========

The Drone
---------
We use the Crazyflie nano quadcopter for our challenge. It is a small, low-cost drone that is easy to program and control. You can find more information about the drone at https://www.bitcraze.io/. Crazyflies have the advantage of being fully open-source and having a large community, which makes it easy to find libraries and tools for programming and controlling the drone. Furthermore, their low price reduces the barrier to entry for participants, and the loss of a drone during the competition is less painful.

The Track
---------
Contrary to other drone racing challenges, this setup not only includes gates in the track but also obstacles that must be avoided by the drones. The current iteration consists of four gates and four obstacles. Gates must be traversed in the correct order and in the correct direction, as passing a gate in the opposite direction will not count as a successful pass. Once a gate has been successfully passed, traversing it again in reverse will not undo the completion; from that point on, the gate simply acts as another obstacle within the track.

All details about the track elements are provided in the level files explained below. The track features two types of gates: tall gates and short gates. The tall gates have a height of 1.195 meters, while the short gates measure 0.695 meters. In both cases, the height is measured from the ground to the center of the gate. Each gate is square in shape, with an outer frame width of 0.72 meters and an inner opening width of 0.4 meters.

The obstacles are designed as slender cylindrical poles with 0.03 meters in diameter. Each obstacle has a total height of 1.52 meters, measured from the ground to the top, and is fitted with a reflective marker mounted on top.

Project Goals
-------------
The goal of the project is to develop a controller that can navigate the drone through the track with the least possible time on a real Crazyflie drone. As previously mentioned, controllers have to be compatible with the predefined interface. Once you have designed a controller and verified its performance and safety in simulation, you can deploy it on the real hardware.

.. warning::
    While the interfaces are the same and we take great care to ensure that the simulation faithfully reproduces the real-world behavior of the drones, running a controller on the real drone is significantly different from running it in simulation. Be aware that unexpected behavior might occur, depending on the actual drone and the used controller, and further extensive tuning and testing may be necessary.

Difficulty Levels
-----------------

The challenge is divided into different difficulty levels, each specified by a TOML configuration file. These levels progressively increase in complexity:

.. list-table::
   :header-rows: 1
   :widths: 10 10 10 10 10

   * - Evaluation Scenario
     - Randomized Inertial Properties
     - Randomized Obstacles, Gates
     - Randomized Tracks
     - Notes
   * - `Level 0 <https://github.com/learnsyslab/lsy_drone_racing/blob/main/config/level0.toml>`_
     - No
     - No
     - No
     - Perfect knowledge
   * - `Level 1 <https://github.com/learnsyslab/lsy_drone_racing/blob/main/config/level1.toml>`_
     - Yes
     - No
     - No
     - Adaptive
   * - `Level 2 <https://github.com/learnsyslab/lsy_drone_racing/blob/main/config/level2.toml>`_
     - Yes
     - Yes
     - No
     - Learning, re-planning
   * - `Level 3 <https://github.com/learnsyslab/lsy_drone_racing/blob/main/config/level3.toml>`_
     - Yes
     - Yes
     - Yes
     - Online planning
   * - sim2real
     - Real hardware
     - Yes
     - Yes
     - Sim2real transfer

..    * - Bonus (`config/multi_level3.toml <https://github.com/learnsyslab/lsy_drone_racing/blob/main/config/multi_level3.toml>`_)
..      - Yes
..      - Yes
..      - Multi-agent racing

On level 0, we have perfect knowledge of the drones properties (mass, inertia) and the track layout. Merely some noise on the action and physics is applied. On level 1, physics parameters (mass, inertia, etc.) or the drone are randomized. At level 2, the position of the gates and obstacles is randomized around their nominal positions. Only when entering the observation range, the exact position is revealed. Finally, at level 3, unlike in level 2 where the nominal positions are known and hard coding a trajectory is possible, the (noisy) positions of gates and obstacles are only known at runtime and the controller must find the optimal path online.

You may use the easier scenarios to develop and debug your controller. However, the final evaluation will be on the hardest scenarios (Level 2 & 3) and the corresponding sim2real transfer.

In real-world deployment, the racing agents are given the nominal positions of the gates and obstacles, before they approach within a certain observation range and receive more accurate measurements, similar to simulation. For level 3, the track layout is generated reversely from randomly placed gates and obstacles in the real world.