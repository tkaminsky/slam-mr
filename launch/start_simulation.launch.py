import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('slam-mr')

    world_file = os.path.join(pkg_share, 'worlds', 'slam-world.world')
    rviz_config = os.path.join(pkg_share, 'config', 'slam_mr.rviz')

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
        env={**os.environ,  'LIBGL_ALWAYS_SOFTWARE': '1'} # type: ignore
    )

    gz_client = ExecuteProcess(
        cmd=['gz', 'sim', '-g'],
        output='screen',
        env={**os.environ,  'LIBGL_ALWAYS_SOFTWARE': '1'} # type: ignore
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription([
        gz_server,
        gz_client,
        rviz_node
    ])