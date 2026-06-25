The Drone Racing Project
========================

Welcome to the LSY Drone Racing Project! This is a platform developed by the LSY Lab at the Technical University of Munich for testing and developing autonomous drone racing algorithms in simulation and deploy the same controller without any modification on real drones.

Implementing Your Own Algorithms
--------------------------------
To implement your own controller, you need to implement a ``Controller`` class in the :mod:`lsy_drone_racing.control` module. The only restriction we place on controllers is that they have to implement the interface defined by the :class:`Controller <lsy_drone_racing.control.controller.Controller>` class. Apart from that, you are encouraged to use the full spectrum of control algorithms, e.g., MPC, trajectory optimization, reinforcement learning, etc., to compete in the challenge. We recommend to put your controller implementation in the :mod:`lsy_drone_racing.control` module to make sure that it is correctly recognized by our scripts.

.. note::
    Make sure to inherit from the base class for your controller implementation. This ensures that your controller is compatible with our scripts. Also, you should only create one controller class per file. Otherwise, we do not know which controller to load from the file.

.. warning::
    You are not allowed to modify the interface of the :class:`Controller <lsy_drone_racing.control.controller.Controller>` class. Doing so will make your controller incompatible with the deployment environment and we won't be able to run your controller on our setup.

.. warning::
    Many students are enthusiastic about deep reinforcement learning and try to use it to solve the challenge. While you are completely free in choosing your control algorithm, we know from experience that training good agents is non-trivial, requires significant compute, and can be difficult to transfer into the real world setup. Students taking this approach should make sure they already have some experience with RL, and take their policies to the real world setup early to address potential sim2real issues. 

Simulation
----------
To get you started, we provide a simulation environment that attempts to reproduce the real-world behavior of the drones as faithfully as possible. You should use this environment to develop and tune your controller. The simulation can also be used to train data-driven controllers that require large amounts of samples, e.g., for reinforcement learning.

Deployment
----------
Once your controller works well in simulation, you can move on to deploying your controller on a real drone. We will provide you with the hardware setup and you can use the same interface to deploy your controller.

Levels
------
The challenge is split into four levels of increasing difficulty. You can find more details in the :doc:`challenge overview <../challenge/overview>`. The final objective is to finish the track on the highest difficulty setting in the shortest time.

Students
--------
Students of the course should make sure to join the Moodle course with the link sent by the instructors. We will give a short introduction to the project in the first and second week of the course. The course is intended for groups of two students. If you do not have a group yet, make sure to look for team partners in our forum.

.. warning::
    Groups should sign up using the form in Moodle during the first week of the course.

Project Structure
-----------------
The project is organized as follows:

The :mod:`lsy_drone_racing` package contains all the code for the project. In :mod:`lsy_drone_racing.control`, we define the interface for the controllers and include example controllers to get you up to speed. The racing environments are implemented in :mod:`lsy_drone_racing.envs`. Deploying the controllers on real drones requires integration with ``ros2`` and the communication library ``cflib`` for our drones. For more details check out the :doc:`Installation and Setup <../getting_started/setup>` section.

The levels are defined in the config files contained in the `config` folder. You can have a look at the configuration options to understand how exactly they differ from each other. 

.. warning::
    Do not modify the config files except for the ``controller.file``, ``deploy`` and ``env.control_mode`` options. You can use these settings to point our scripts for simulation and deployment to your controller implementation, to select the control mode of the environment, and to configure the deployment settings.

We also provide scripts to simulate and deploy your controller. You can find them in the `scripts` folder. A more detailed description of how to simulate and deploy your controller can be found in the challenge description of the :doc:`simulation <../challenge/simulation>` and :doc:`deployment <../challenge/deployment>`.
