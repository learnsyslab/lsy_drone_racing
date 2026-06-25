Simulation
==========

The first part of the challenge is to complete the race track in simulation. To that end, we provide a high-fidelity simulator with interfaces that are identical to the real-world setup. We use `crazyflow <https://github.com/learnsyslab/crazyflow/tree/main>`_ as the underlying simulator to model the physics of the drone and to render the environment.

Running the Simulation
----------------------

Once you have installed the ``lsy_drone_racing`` package, you should be able to start a simulation run with

.. code-block:: bash

   python scripts/sim.py

The simulation script has a built-in CLI that allows you to set a few useful arguments. For example, if you want the simulation to render its output and run more than just a single race, you can pass these arguments with

.. code-block:: bash

   python scripts/sim.py -r -n 10

This will render the simulation (``-r`` or ``--render``) and run 10 races in succession (``-n`` or ``--n_runs``). For a list of all arguments, use ``-h`` or ``--help``.


Switching Between Configurations
--------------------------------

You can choose which configuration to use by changing the ``--config`` command line option. For example, to run the example controller on the hardest scenario, use the following command:

.. code-block:: bash

   python scripts/sim.py --config level3.toml

To use your own controller, you can pass the path to your controller script as the ``--controller`` argument. For example:

.. code-block:: bash

   python scripts/sim.py --config level3.toml --controller my_controller.py

.. note::
    You can also write your controller file directly into the config files located in the ``config`` folder. That way, you don't need to specify the controller script when running the simulation, and the controller will be used in the automated challenge evaluation.