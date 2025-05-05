# SLAM-MR

## Group 36: Thomas Kaminsky, Hammad Izhar

Below is a summary of our final project for CS2620: Distributed Computing.

For this project, we were interested in implementing multi-agent SLAM algorithms. In particular, since the scope of SLAM is so large, we wanted to focus on the implementation of backend SLAM algorithms for pose estimation. Commonly, these algorithms take the form of pose-graph optimization (PGO). In this repository we provide a (semi)-reimplementation of the distributed PGO procedure described in [Choudhary et al. 2017](https://arxiv.org/abs/1702.03435) using successive over-relaxation (SOR).

Unlike existing reference implementations that we could find, we wanted to perform all of the inter-robot communication using ROS. In doing so, we have achieved the following:

1. A working distributed implementation of the algorithm described in [Choudhary et al. 2017](https://arxiv.org/abs/1702.03435) (see `slam-mr/slam_robot_node.py`) using ROS for communication
2. A Gazebo simulation capable of supporting multiple robots equipped with LIDAR sensors. The sensed point clouds are bridged over to ROS as `sensor_msg/msg/LaserScans`.
3. An RViz visualization which plugs into the local odometry measurements and laser scans to visualize the sensed maps.

## Setup and Running

This entire repository was developed on M2 Macbook Pros with the help of [Robostack](https://robostack.github.io/GettingStarted.html#__tabbed_1_1) to simplify setting up the ROS2 Jazzy environment. We defer to the Robostack documentation for how to setup ROS2 and install Gazebo Harmonic, noting that in most cases once Robostack has been successfully configured one can follow any Linux tutorial and replace all `sudo apt install <pkg_name>` commands with the equivalent `mamba install <pkg_name>` command.

Once ROS2 and Gazebo are installed, building the repository is as simple as creating a new ROS2 workspace. We recommend the following:

```
mkdir -p slam_ws/src
cd slam_ws/src
git clone https://github.com/Hammad-Izhar/slam_mr_msgs
git clone https://github.com/tkaminsky/slam-mr
cd ..
colcon build --symlink-install
```

It is expected for the messages to take slightly longer to build than the actual code repository. Furthermore, on MacOS there are warnings in stderr that can safely be ignored.

Once the colcon workspace has been built the simulation can be launched using the `start_simulation.launch.py` launch file.

```
ros2 launch slam-mr start_simulation.launch.py
```

Ground robots can be spawned into the simulation using the `spawn_ground_robot.launch.py`:

```
ros2 launch slam-mr spawn_ground_robot.launch.py namespace:=<robot_id> x:=<x> y:=<y>
```

We can launch the distributed node implementation using

```
ros2 run slam-mr slam_robot_node
```

Details about our implementation can be found in our written report, `final_writeup.pdf`.
