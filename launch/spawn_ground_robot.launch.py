import os
import xacro
import tempfile
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from ros_gz_bridge.actions import RosGzBridge


def create_robot_pipeline(context, *args, **kwargs):
    # get the namespace for this robot
    namespace = context.launch_configurations['namespace']
    pkg_share = get_package_share_directory('slam-mr')

    # process Xacro
    xacro_file = os.path.join(pkg_share, 'urdf', 'ground_robot.urdf.xacro')
    xacro_doc = xacro.process_file(xacro_file, mappings={'namespace': namespace})  # override namespace
    robot_description = xacro_doc.toxml()

    # write a temporary bridge YAML for this namespace
    bridge_cfg = [
            {
                'gz_topic_name': f'/model/{namespace}/cmd_vel',
                'ros_topic_name': 'cmd_vel',
                'ros_type_name': 'geometry_msgs/msg/Twist',
                'gz_type_name': 'gz.msgs.Twist',
            },
            {
                'gz_topic_name': f'/model/{namespace}/odometry',
                'ros_topic_name': 'odom',
                'ros_type_name': 'nav_msgs/msg/Odometry',
                'gz_type_name': 'gz.msgs.Odometry',
            },
            {
                'gz_topic_name': f'/model/{namespace}/scan',
                'ros_type_name': 'sensor_msgs/msg/LaserScan',
                'ros_topic_name': 'scan',
                'ros_type_name': 'sensor_msgs/msg/LaserScan',
                'gz_type_name': 'gz.msgs.LaserScan',
            },
            {
                'gz_topic_name': f'/model/{namespace}/tf',
                'ros_topic_name': 'tf',
                'ros_type_name': 'tf2_msgs/msg/TFMessage',
                'gz_type_name': 'gz.msgs.Pose_V',
            },
    ]

    tmp_cfg = os.path.join(tempfile.gettempdir(), f'bridge_{namespace}.yaml')
    with open(tmp_cfg, 'w') as f:
        yaml.safe_dump(bridge_cfg, f)

    # build actions
    actions = []
    # push namespace
    actions.append(PushRosNamespace(namespace))

    # robot_state_publisher
    actions.append(
        Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            name='robot_state_publisher', output='screen',
            parameters=[{'robot_description': robot_description}]
        )
    )

    # spawn robot in Gazebo
    actions.append(
        Node(
            package='ros_gz_sim', executable='create',
            name='spawn_robot', output='screen',
            arguments=['-topic', 'robot_description', '-name', namespace, '-x', '0', '-y', '0', '-z', '0.1']
        )
    )

    # bridge ROS <-> Gazebo topics
    actions.append(
        RosGzBridge(
            bridge_name=f'bridge_{namespace}',
            config_file=tmp_cfg
        )
    )

    # teleop keyboard node
    actions.append(
        Node(
            package='teleop_twist_keyboard', executable='teleop_twist_keyboard',
            name='teleop_twist_keyboard', output='screen', prefix="xterm -e",
            remappings=[('/cmd_vel', 'cmd_vel')]
        )
    )

    return actions


def generate_launch_description():
    namespace_arg = DeclareLaunchArgument(
        'namespace', default_value='ground_robot',
        description='Namespace and Gazebo entity name'
    )

    return LaunchDescription([
        namespace_arg,
        OpaqueFunction(function=create_robot_pipeline)
    ])
