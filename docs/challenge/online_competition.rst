Online Competition
==================

The online competition takes place every semester for the students of the "Autonomous Drone Racing" class at TUM. Students compete against each other on :doc:`level 2 <../overview>`. 

.. note::
    The competition is useful to get an idea of how fast your controller is compared to other teams. It is **not** part of your grade or used to evaluate your final project submission.

Leaderboard
-----------
We host the online competition directly on GitHub in the `competition branch <https://github.com/learnsyslab/lsy_drone_racing/tree/competition>`_. Students can submit their controllers to the competition by forking the repository and pushing their changes to their main branch. The fastest submission with at least 50% success rate will be used for the team ranking in the online leaderboard.

.. note::
    The leaderboard is reset every semester. Participation is by whitelist only. If you have not been whitelisted as a student in the course, please contact the instructors.

Signing Up
----------
To participate in the online competition, follow these steps:

#. Fork the lsy_drone_racing repository to your own GitHub account.

#. Add the provided DISPATCH_TOKEN to your fork's secrets under Settings > Secrets and variables > Actions

Submitting Your Controller
--------------------------
After forking the repository, you can start implementing your controller. For this, we advise you to work on a separate feature branch on your fork. We provide several controller examples in the control module. Once you have implemented your own controller, you should set the ``controller.file`` argument in the level 2 config file to your controller file. This will ensure that the controller is used by default when running the evaluation script. Before submission, you can evaluate your controller locally by running

.. code-block:: bash

    pixi run evaluate
    # or
    python scripts/evaluate.py

When your controller passes the evaluation and you want to submit your controller to the leaderboard, you need to push your changes to the main branch of your forked repository. This will trigger a GitHub action on our central repository to evaluate your controller and add the results to the leaderboard.

.. note::
    Before evaluation, the server will check your code for cheats. In the ``level2.toml`` file, you may only change the path to your controller under ``controller.file`` and the used control interface under ``env.control_mode``. Your submission will be rejected if any other changes are detected.

If you need extra dependencies to run your controller, you can add them to the default environment in the ``pyproject.toml`` or by using ``pixi add <dependency>``. This is necessary so the repo can execute your code.

.. note::
    After changing dependencies, make sure your lock file (``pixi.lock``) is up to date. Otherwise, the worker won't be able to run your code. To update your lock file, you may use ``pixi install`` or start the changed environments.

.. note::
    The evaluation takes a few minutes on the worker. Please evaluate your solution locally on another branch before pushing to main and therefore submitting to the competition.

For a more detailed analysis of your performance, you can review the test output in the GitHub action logs. This information can be valuable for fine-tuning your controller and improving your results.
