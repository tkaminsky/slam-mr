import os
import xacro

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from ros_gz_bridge.actions import RosGzBridge
from launch_ros.actions import Node


def generate_launch_description():
    # Include the simulation & robot spawn
    pkg_share = get_package_share_directory('slam-mr')

    xacro_file_path = os.path.join(pkg_share, 'urdf', 'ground_robot.urdf.xacro')
    bridge_yaml = os.path.join(pkg_share, 'config', 'bridge.yaml')

    xacro_doc = xacro.process_file(xacro_file_path) # type: ignore
    robot_description_xml = xacro_doc.toxml() # type: ignore

    # Publish robot_description directly
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_xml}], # type: ignore
    )

    spawn_robot = Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_ground_robot',
            arguments=[
                '-topic', '/robot_description',
                '-entity', 'ground_robot',
                '-x', '0.0', '-y', '0.0', '-z', '0.1',
            ],
            output='screen',
        )

    ros_gz_bridge = RosGzBridge(
        bridge_name='ros_gz_bridge',
        config_file=bridge_yaml,
    )

    # # Keyboard teleop: publishes geometry_msgs/Twist to /cmd_vel
    teleop = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'teleop_twist_keyboard', 'teleop_twist_keyboard',
            '--ros-args',
            '--remap', '/cmd_vel:=/cmd_vel'
        ],
        output='screen',
        shell=False
    )

    return LaunchDescription([
        robot_state_publisher,
        spawn_robot,
        ros_gz_bridge,
        # teleop
      ])
