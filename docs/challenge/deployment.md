# Deployment

The idea for the deployment is to have an environment that exactly matches the interfaces and observation space of the simulation. Since the action space for the controller is identical, controllers can be directly deployed on the real drone without any modifications.

!!! warning
    Please be aware that running a controller on the real drone may still exhibit significant differences compared to the simulation due to the sim2real gap.

## Motion Tracking

We use a [Vicon](https://www.vicon.com/) motion tracking system to track the motion of the drone. The Vicon system consists of several cameras that are placed around the track, and a base station that calculates object poses by triangulation. Gates, obstacles and the drone are all equipped with reflective markers, which can be tracked by the cameras. Since we'd need to resort to numerical differentiation to get velocity information, we're running state estimators that filter the noisy Vicon measurements and provide smoother estimates of the drone's state.

After completing the [hardware setup](../getting_started/setup.md#hardware-setup-ubuntu-only), you need a total of *three* open terminals in the repository to deploy your controller. The commands below use the deploy environment.

In the first terminal, run this command to launch the translation layer from the motion capture cameras to ROS2.

```bash
pixi run -e deploy mocap
```

!!! warning
    If you cannot see the drone in RVIZ, it is likely that Vicon is not turned on, or the drone is not selected for tracking in the Vicon system.

The second terminal is used to launch the estimator for the drone. If you want to use the default settings, it's enough to specify your drone ID with `pixi run -e deploy estimator <drone_name>`. You need to use the actual DEC number on the drone or the name shown in rviz. If the estimator works, you should see the frequency information in the terminal.

```bash
pixi run -e deploy estimator cf01
```

## Setting up the Track 
We provide a script that can help you set up the track. It will publish the nominal positions and orientations of the gates, obstacles and drones to RVIZ as MarkerArray.
```bash
pixi shell -e deploy
python scripts/nominal_frame_publisher.py --config level2.toml
```  
Now you need to add the MarkerArray to RVIZ as follows:
<div align="center">
  <img src="../../img/rviz_setup.png" alt="RVIZ SETUP" width="600"/>
</div>

## Deploying Your Controller

To deploy your controller on the real drone, use the deployment script in the `lsy_drone_racing/scripts` folder. Place the drone on its start position, power it on, and launch the estimators.

!!! note
    Make sure the drone has enough battery to complete the track. If a red LED is constantly turned on, the drone is low on battery. A blinking red LED indicates that the battery is sufficiently charged.

In the third terminal, start the deploy environment with `pixi shell -e deploy` and the deployment script with the correct configuration and controller.

```bash
pixi shell -e deploy
python scripts/deploy.py
# or
python scripts/deploy.py --config <config_name>.toml --controller <controller_name>.py
```

!!! note
    Be careful when flying the drone! Make sure to kill the process (**Ctrl+C**) immediately when your controller is unstable.

The deployment script will first check if the real track poses and the drone starting pose is within acceptable bounds of the configured track. If not, the script will print an error message and terminate. If the poses are correct, the drone will take off, fly through the track, print out the final lap time, and land automatically.

## Saving a Track Layout (Optional)

If you want to save the measured layout for later use, run the `save_track_as_config.py` script in the `lsy_drone_racing/scripts` folder:

```bash
pixi shell -e deploy
python scripts/save_track_as_config.py --config <config_name>.toml --save_config_to <output_name>.toml
```

The script saves the measured gate, obstacle, and drone starting poses to `config/<output_name>.toml`.
