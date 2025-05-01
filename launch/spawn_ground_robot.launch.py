import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ros_gz_bridge.actions import RosGzBridge

def generate_launch_description():
    pkg_share = get_package_share_directory('slam-mr')

    # Paths to world and xacro files
    world_file = os.path.join(pkg_share, 'worlds', 'world.world')
    xacro_file_path = os.path.join(pkg_share, 'urdf', 'ground_robot.urdf.xacro')
    bridge_yaml = os.path.join(pkg_share, 'config', 'bridge.yaml')

    # Process Xacro to URDF XML string
    xacro_doc = xacro.process_file(xacro_file_path) # type: ignore
    robot_description_xml = xacro_doc.toxml() # type: ignore

    gpu_workaround_env = {
        'LIBGL_ALWAYS_SOFTWARE': '1',
    }
    merged_env = os.environ.copy()
    merged_env.update(gpu_workaround_env)

    # Start Gazebo server and client
    gz_server = ExecuteProcess(
        cmd=[
            'gz', 'sim',
            '-s',                              # server only
            '-r',                              # run immediately
            '-v', '4',                         # verbose
            '--render-engine', 'ogre2',
            world_file
        ],
        output='screen',
        env=merged_env # type: ignore
    )

    gz_client = ExecuteProcess(
        cmd=['gz', 'sim', '-g'],
        output='screen',
        env=merged_env # type: ignore
    )

    # Publish robot_description directly
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_xml}], # type: ignore
    )

    spawn_robot = TimerAction(
        period=2.0,
        actions=[Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_ground_robot',
            arguments=[
                '-topic', '/robot_description',
                '-entity', 'ground_robot',
                '-x', '0.0', '-y', '0.0', '-z', '0.1',
            ],
            output='screen',
        )]
    )

    ros_gz_bridge = RosGzBridge(
        bridge_name='ros_gz_bridge',
        config_file=bridge_yaml,
    )


    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # Return assembled LaunchDescription
    return LaunchDescription([
        gz_server,
        gz_client,
        robot_state_publisher,
        spawn_robot,
        ros_gz_bridge,
        # rviz_node
    ])
