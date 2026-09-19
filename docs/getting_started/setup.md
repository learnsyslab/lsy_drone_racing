# Installation and Setup

This guide will walk you through the process of setting up the LSY Autonomous Drone Racing project on your system.

## Prerequisites

Before you begin, ensure you have the following:

- [Git](https://git-scm.com/install/) installed on your system
- A [GitHub](https://github.com/) account
- A [Robostack](https://robostack.github.io/index.html/) environment with [pixi running ROS2 Jazzy](https://robostack.github.io/GettingStarted.html#__tabbed_1_3/)（automatically installed during the deployment setup below）

!!! note
    You can also use [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) or other dependency management tools, but we won't cover them here in full detail.

## Required Repositories

The LSY Autonomous Drone Racing project requires you to fork the drone racing repository:

[lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing) (main branch)

This repository contains the drone simulation, environments, and scripts to simulate and deploy the drones in the racing challenge.

Depending on if you want to use the simulation only or also deploy on real drones, you need additional dependencies. Note that you don't have to install any of those manually, since our setup scripts will take care of that.

### In Simulation

- [lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing) – environments and scripts for simulation and deployment
- [crazyflow](https://github.com/learnsyslab/crazyflow) – drone simulator
- [drone-models](https://github.com/learnsyslab/drone-models) – Crazyflie dynamics models
- [drone-controllers](https://github.com/learnsyslab/drone-controllers) – controller implementations

### On Hardware

To run the project on real drones, add:

- [motion_capture_tracking](https://github.com/learnsyslab/motion_capture_tracking) – publishes motion capture data to ROS2
- [drone-estimators](https://github.com/learnsyslab/drone-estimators) – drone state estimators

## Step-by-Step Installation

### Fork lsy_drone_racing

Start by forking the [lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing) repository for your own group. This serves two purposes:

1. You'll have your own repository with git version control and automated testing.
2. It sets you up for participating in the [online competition](../challenge/online_competition.md).

If you're new to GitHub, refer to the [GitHub documentation on forking](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).

### Setting up your environment

#### Pixi package manager (Recommended)

We recommend using [Pixi](https://pixi.sh) to manage dependencies  and virtual environments for this project. Pixi creates a dedicated .pixi directory in the project root, which contains the isolated virtual environment. We use Pixi in combination with RoboStack. Robostack lets you install your favorite ROS version independent of your OS. Installed packages are cached globally under ~/.cache/rattler/ to speed up subsequent environment setups across projects.

Install Pixi:

=== "Linux & macOS"

    ```bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
    ```
To activate the environment, simply run

```bash
pixi shell
# or
pixi shell -e <environment_name>
```

!!! note
    To leave a pixi shell, run `exit` or press **Ctrl+D**. Make sure to leave the shell before you activate another shell.

On the first invocation, Pixi will automatically resolve and install all required dependencies.

!!! note
    In the pixi shell, you can still use pip to install any packages you need.

#### Micromamba package manager (Not recommended)

You may also use  with [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). We do not recommend this, since Mamba/Conda environments have the tendency to be leaky and share some system-wide packages. In our experience, this will lead to problems, which is why this project is optimized to use with pixi. If you want to use micromamba anyway, we can't guarantee support.

#### Simulation & Hardware on our Lab PC 

We provide a Workstation in the Lab on which you are allowed to run your controllers during deployment. Please create a new user for each team and follow the instructions below.

#### Simulation only (Not Recommended)

If you only want to run the simulation, you can use your favorite conda/mamba/venv to install the packages. However, you will need a working installation of ROS2 if you want to deploy your controller on the real drone.

### Installation

#### Supported platforms and environments

| Pixi environment | Purpose | Ubuntu (x86-64) | macOS (Apple Silicon) | Windows (x86-64, native) |
| --- | --- | --- | --- | --- |
| `default` | CPU simulation and controller development | Yes | Yes | Yes |
| `deploy` | Real-drone deployment with ROS 2 | Yes | No | No |
| `gpu` | CUDA simulation | Yes | No | No |
| `tests` | Tests | Yes | Yes | Yes |
| `gpu-tests` | Tests with CUDA | Yes | No | No |
| `docs` | Build and preview documentation | Yes | Yes | Yes |

Use `default` for simulation and `deploy` for real-drone deployment. Follow the installation steps below to set up the appropriate environment.

#### Clone repository

First, clone your fork from your own account and create a new environment by running

=== "Linux & macOS"

    ```bash
    mkdir -p ~/repos && cd repos
    git clone https://github.com/<YOUR-USERNAME>/lsy_drone_racing.git
    cd lsy_drone_racing
    ```

=== "Windows"

    ```powershell
    mkdir "$HOME\repos" -Force
    cd "$HOME\repos"
    git clone https://github.com/<YOUR-USERNAME>/lsy_drone_racing.git
    cd lsy_drone_racing

#### Install simulation environment (developing & testing controllers)

Stay in the repository and run the following command to activate your pixi shell with the default (sim) environment:

```bash
pixi shell
```

!!! note
    Some subpackages currently depend on a prerelease version of [scipy](https://github.com/scipy/scipy), which needs to be built from source. This might take more than 10 minutes on older hardware.

!!! note
    By running the commands above, our automated scripts will install and activate **acados** by default. This might cause the terminal to freeze for several minutes. [Acados](https://docs.acados.org/index.html) is an Optimal Control Framework that can be used to control the quadrotor using a Model Predictive Controller. If something does not work out of the box, we refer the reader to the [official installation guide](https://docs.acados.org/installation/).

(Optional, Ubuntu Only)To speed up simulation with GPU , run:

```bash
pixi shell -e gpu
```

Finally, you can test if the installation was successful by running

```bash
cd ~/repos/lsy_drone_racing
python scripts/sim.py
```

If everything is installed correctly, this opens the simulator and simulates a drone flying through four gates.

(Optional, Ubuntu Only) If you want to train RL policies, we recommend using a GPU-enabled environment for optimal performance. To install additional dependencies including [PyTorch](https://pytorch.org/) and [Wandb](https://wandb.ai/), stay in the gpu shell and run:

```bash
pip install -e .[rl]
```

(Optional) You can also run the tests by directly running either

```bash
pixi run -e tests tests -v
```

or by first activating the correct environment

```bash
pixi shell -e tests
cd ~/repos/lsy_drone_racing
pytest tests
```

#### Install deployment environment (deploy controller to real drones, Ubuntu Only)

This is for the deployment in the lab, either on your own machine or on the lab PC. With a fresh terminal, stay in the repository and run:

```bash
pixi shell -e deploy
```

This will automatically create a ros2 workspace with RoboStack, clone the motion_capture_tracking package, and build it. Thus, the terminal might freeze for 1 minute after regular installation.

!!! note
    By running the commands above, our automated scripts will install and activate **acados** by default. This might cause the terminal to freeze for several minutes. [Acados installation guide](https://docs.acados.org/index.html) is an Optimal Control Framework that can be used to control the quadrotor using a Model Predictive Controller. If something does not work out of the box, we refer the reader to the [official installation guide](https://docs.acados.org/installation/).

Test your installation: For this to work you have to be in the lab and be connected to our local network.

```bash
ping 10.157.163.191
```

If this works, you need a total of *three* open terminals with the deploy environment activated to actually deploy your controller. Before we do that, however, we need to prepare the USB port for the Crazyradio to send commands to the Crazyflie drones. For that, execute the following block. Ask the TA to help you with sudo rights.

```bash
cat <<EOF | sudo tee /etc/udev/rules.d/99-bitcraze.rules > /dev/null
# Crazyradio (normal operation)
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="7777", MODE="0664", GROUP="plugdev"
# Bootloader
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="0101", MODE="0664", GROUP="plugdev"
# Crazyflie (over USB)
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0664", GROUP="plugdev"
EOF

# USB preparation for crazyradio
sudo groupadd plugdev
sudo usermod -a -G plugdev YOUR-USERNAME

# Apply changes
sudo udevadm control --reload-rules
sudo udevadm trigger
```

!!! note
    For TAs: Switch to the admin account using `su` and replace
    `YOUR-USERNAME` with the student's username.

Now you are ready to deploy your controller on real drones. First, run the motion capture tracking node. If there are valid elements in the motion capture area, you should be able to see them in the rviz window.

```bash
pixi run mocap
```

Second, start another deploy shell and run the estimator node. Please check the actual DEC number on the drone, or the name shown in rviz. If this works, you should be able to see frequency information in terminal.

```bash
pixi run estimator cf01
```

Lastly, run the deployment script with the correct configuration and controller.

```bash
python scripts/deploy.py --config level2.toml --controller <your_controller.py>
```

!!! note
    Be careful when flying the drone! Make sure to kill the process (**Ctrl+C**) immediately when your controller is unstable.

#### Work on Existing Dependencies

If you want to do more in-depth development or understand the used packages ([crazyflow](https://github.com/learnsyslab/crazyflow), [drone-models](https://github.com/learnsyslab/drone-models), [drone-controllers](https://github.com/learnsyslab/drone-controllers), [drone-estimators](https://github.com/learnsyslab/drone-estimators)) better, you can fork and install all of those packages separately in editable mode. If you find bugs or other have improvements, feel free to [submit a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) or [create an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue) to help us improve the code. The installation procedure is the same for all packages:

```bash
cd ~/repos/lsy_drone_racing
pixi shell
cd ~/repos
git clone https://github.com/<YOUR-USERNAME>/crazyflow.git
cd ~/repos/crazyflow
pip install -e .
cd ~/repos/lsy_drone_racing
```

After installation, you can change files in the cloned repository and the changes will directly affect your environment.

#### Extended Dependencies

We want to encourage you to use other libraries to speed up your development process. The easiest way to use another library is to install it with pip inside your pixi shell.

!!! warning
    If your controller depends on additional libraries, which are installed locally with pip, the tests and evaluation on GitHub won't work.

To properly add a package to your project, you can either add it to the `pyproject.toml` file in the root of the repository, or run the following command while being in the correct pixi environment:

```bash
pixi add <package_name>
```

After that, reopen your environment. This automatically adds the package to the `pyproject.toml` file.

!!! note
    Changing the `pyproject.toml` will also update the `pixi.lock` file, which pins the exact versions of all packages. Make sure to commit both files to your repository, otherwise the tests on GitHub will fail.

## Common errors

### LIBUSB_ERROR_ACCESS (deployment only)

If you encounter USB access permission issues, change the permissions with the following command. You might need help from a TA to get sudo rights.

```bash
sudo chmod -R 777 /dev/bus/usb/
```

### Drone won't start (deployment only)

Usually, the error messages should give you a good idea of what is going wrong. If you have no idea, check the following before asking your TA:

1. Make sure your drone is selected on the vicon system, otherwise it will not be tracked.

2. Make sure your drone is visible in rviz. If you cannot see the drone in RVIZ, it is likely that Vicon is not turned on, or the drone is not selected for tracking in the Vicon system.

3. Make sure the estimator process is running and outputting frequency information, otherwise the process is not running properly.

4. Make sure you have selected the correct drone id and channel in the config file.

### libdecor Warning

If you encounter warnings like

```text
libdecor-gtk-WARNING: Failed to initialize GTK
Failed to load plugin 'libdecor-gtk.so': failed to init
No plugins found, falling back on no decorations
```

Note that starting the simulation with `-r` from a terminal inside VSCode might cause this warning. This will cause your window to not have any decorations (close, minimize, maximize buttons). You can safely ignore this warning. If you want to get rid of it, start the simulation from a regular terminal outside of VSCode.

### PowerShell script execution is disabled (Windows only)

If `pixi shell` fails with an error like:

```text
File ...\Temp\tmp....ps1 cannot be loaded because running scripts is disabled on this system.
```

PowerShell's `Restricted` execution policy blocks the temporary script that Pixi uses
to activate the environment. Check the policy and allow local scripts for the current
session, then retry:

```powershell
Get-ExecutionPolicy
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
pixi shell
```

This change requires no administrator privileges and applies only to the current
PowerShell session and its child processes. It expires when they are closed and does
not permanently change the user or system execution policy.

## Next Steps

Once you have successfully set up the project, you can proceed to explore the simulation environment, develop your racing algorithms, and participate in the online competition. Refer to other sections of the documentation for more information on using the project and developing your racing strategies.
