LSY Drone Racing
================

Welcome to the LSY Drone Racing documentation! This project provides a simulator and code for deployment on real robot hardware for autonomous drone racing.

.. image:: img/race.gif
   :alt: LSY Drone Racing
   :align: center
   :width: 600px


Project Overview
----------------

LSY Drone Racing is designed to help you develop and test autonomous drone racing algorithms. Whether you're a beginner or an experienced drone enthusiast, this documentation will guide you through the process of setting up your environment, understanding the challenge, and utilizing the provided Python API to create high-performance racing drones.

Key Features:
^^^^^^^^^^^^^

1. **Simulation Environment**: Test your algorithms in a realistic virtual environment before deploying to real drones.
2. **Deployment Tools**: Transition from simulation to real-world racing with interfaces compatible with the simulation environment.
3. **Online Competition**: Participate in virtual drone racing competitions to benchmark your algorithms against others.
4. **Comprehensive Python API**: Access a comprehensive documentation of the Python API to get a deeper understanding of how the simulation and deployment environments work under the hood.

Getting Started
^^^^^^^^^^^^^^^

If you're new to the project, start with the :doc:`Getting Started <getting_started/general>` section. This will help you set up your environment and understand the basics of the LSY Drone Racing framework.

The Challenge
^^^^^^^^^^^^^

Learn about the :doc:`drone racing challenge <challenge/overview>`, including details on the :doc:`simulation environment <challenge/simulation>`, :doc:`real-world deployment <challenge/deployment>`, and the :doc:`online competition <challenge/online_competition>`.

Python API
^^^^^^^^^^

If you want to understand the ins and outs of the framework, you can explore the extensive Python API, which includes modules for:

- :doc:`Drone Control <control/index>`
- :doc:`Racing Environments <envs/index>`
- :doc:`Utility Functions <utils/index>`

We hope this documentation helps you dive into the exciting world of autonomous drone racing. Good luck, and may the best algorithm win!

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   getting_started/general
   getting_started/setup

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: The Challenge

   challenge/overview
   challenge/simulation
   challenge/deployment
   challenge/online_competition

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Python API

   control/index

   envs/index

   utils/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
