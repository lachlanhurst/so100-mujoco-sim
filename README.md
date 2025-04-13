# so100 MuJoCo Simulation User Interface
User interface to simulate and drive the so100(so-arm100) robot arm. It supports the following:
- Control the arm (real and simulation) by independent joint angles in the user interface
- Control the simulation by moving the real robot
- Record the joint angles as the arm is controlled using one of the above methods
- Playback the recorded joint angle on the arm (real and simulation)

Ideally you have a so100 robot arm connected, but the application will still work fine without one.


## Dependencies

This project uses the [Pixi](https://pixi.sh/) package management tool, you will need to install this.

The application uses [MuJoCo](https://mujoco.org/) for simulation and visualisation of the robot arm. [PySide6](https://doc.qt.io/qtforpython-6/) (Qt) is used as the widget toolkit. Both these are installed during the pixi based getting started process below.

[LeRobot](https://github.com/huggingface/lerobot) is used to control the robot and also for the calibration process. LeRobot code is downloaded in the `pixi run download` step below.


## Getting started

Clone the repo

    git clone https://github.com/lachlanhurst/so100-mujoco-sim.git
    cd so100-mujoco-sim

Install dependencies

    pixi install

Download the so100 MuJoCo xml files from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). This may take a little while as it downloads all models, then extracts ony the so100.

    pixi run download

Run the simulation UI

    pixi run simulate


## Running tests

Unit tests can be run with the following command

   pixi run tests


# Acknowledgements

Thanks to [TheRobotStudio for making such a great robot](https://github.com/TheRobotStudio/SO-ARM100)!
