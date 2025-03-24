# so100 MuJoCo Simulation and UI
Simple UI to simulate and drive the so100(so-arm100) robot arm.


## Dependencies

This project uses the [Pixi](https://pixi.sh/) package management tool, you will need to install this.

The application uses [MuJoCo](https://mujoco.org/) for simulation and visualisation of the robot arm. [PySide6](https://doc.qt.io/qtforpython-6/) (Qt) is used as the widget toolkit. Both these are installed during the pixi based getting started process below.


## Getting started

Clone the repo

    git clone https://github.com/lachlanhurst/so100-mujoco-sim.git
    cd so100-mujoco-sim

Install dependencies

    pixi install

Download the so100 urdf files from [the source repo](https://github.com/TheRobotStudio/SO-ARM100). This will download a complete copy of the urdf files, including STL meshes used by the mujoco model.

    pixi run download

Run the simulation UI

    pixi run simulate
