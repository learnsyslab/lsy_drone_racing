Deployment
==========

The idea for the deployment is to have an environment that exactly matches the interfaces and observation space of the simulation. Since the action space for the controller is identical, controllers can be directly deployed on the real drone without any modifications.

.. warning::
    Please be aware that running a controller on the real drone may still exhibit significant differences compared to the simulation due to the sim2real gap.

Motion Tracking
---------------
We use a `Vicon <https://www.vicon.com/>`_ motion tracking system to track the motion of the drone. The Vicon system consists of several cameras that are placed around the track, and a base station that calculates object poses by triangulation. Gates, obstacles and the drone are all equipped with reflective markers, which can be tracked by the cameras. Since we'd need to resort to numerical differentiation to get velocity information, we're running state estimators that filter the noisy Vicon measurements and provide smoother estimates of the drone's state.

As mentioned in the :doc:`Installation and Setup <../getting_started/setup>` section, you need to run two terminals to launch a ROS2 node with the ``motion_capture_tracking`` package that make the Vicon poses available:

.. code-block:: bash

    pixi shell -e deploy
    ros2 launch motion_capture_tracking launch.py

.. warning::
    If you cannot see the drone in RVIZ, it is likely that Vicon is not turned on, or the drone is not selected for tracking in the Vicon system.

The second terminal is used to launch the estimator for the drone. If you want to use the default settings, it's enough to specify your drone ID with the ``--drone_name`` argument. For advanced settings, you need to modify the ``estimators.toml`` file in the ``drone-estimators`` repository or pass a path to your TOML file with the ``--settings`` argument.

.. code-block:: bash

    pixi shell -e deploy
    python -m drone_estimators.ros_nodes.ros2_node --drone_name cf52

Generating Tracks for Level 3 Deployment
------------------------------------------
For level 3 deployment, you need to generate new configuration file with a track layout based on the real-world positions of the gates and obstacles. To do so, use the ``save_track_as_config.py`` script in the ``lsy_drone_racing/scripts`` folder. 

.. code-block:: bash

    python scripts/save_track_as_config.py --config <config_name> --save_config_to <config_output_path>

The script will query the Vicon system for the current positions of all gates and obstacles, and generate a new deploy-ready TOML configuration file with the specified name.

Deploying Your Controller
-------------------------
To deploy your controller on the real drone, use the deployment script in the ``lsy_drone_racing/scripts`` folder. Place the drone on its start position, power it on, and launch the estimators.

.. note::
    Make sure the drone has enough battery to complete the track. If a red LED is constantly turned on, the drone is low on battery. A blinking red LED indicates that the battery is sufficiently charged.

.. code-block:: bash

    python scripts/deploy.py --config <config_name> --controller <controller_name>

The deployment script will first check if the real track poses and the drone starting pose is within acceptable bounds of the configured track. If not, the script will print an error message and terminate. If the poses are correct, the drone will take off, fly through the track, print out the final lap time, and land automatically.
