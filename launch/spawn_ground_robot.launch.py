import os
import xacro
import tempfile
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext, Action
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node, PushRosNamespace

def create_robot_pipeline(context: LaunchContext) -> list[Action]:
    pkg_share = get_package_share_directory('slam-mr')

    namespace = context.launch_configurations['namespace']
    x_offset = context.launch_configurations['x']
    y_offset = context.launch_configurations['y']

    # process Xacro
    xacro_file = os.path.join(pkg_share, 'urdf', 'ground_robot.urdf.xacro')
    xacro_doc = xacro.process_file(xacro_file, mappings={'namespace': namespace}) # type: ignore
    robot_description = xacro_doc.toxml() # type: ignore

    # write a temporary bridge YAML for this namespace
    bridge_cfg = [
            {
                'gz_topic_name': f'/model/{namespace}/cmd_vel',
                'gz_type_name': 'gz.msgs.Twist',
                'ros_topic_name': 'cmd_vel',
                'ros_type_name': 'geometry_msgs/msg/Twist',
                'direction': 'ROS_TO_GZ'
            },
            {
                'gz_topic_name': f'/model/{namespace}/odometry',
                'gz_type_name': 'gz.msgs.Odometry',
                'ros_topic_name': 'odometry',
                'ros_type_name': 'nav_msgs/msg/Odometry',
                'direction': 'GZ_TO_ROS'
            },
            {
                'gz_topic_name': f"/model/{namespace}/pose",
                'gz_type_name': 'gz.msgs.Pose',
                'ros_topic_name': 'pose',
                'ros_type_name': 'geometry_msgs/msg/Pose',
                'direction': 'GZ_TO_ROS'
            },
            # {
            #     'gz_topic_name': f"/model/{namespace}/pose",
            #     'gz_type_name': 'gz.msgs.Pose',
            #     'ros_topic_name': '/tf',
            #     'ros_type_name': 'tf2_msgs/msg/TFMessage',
            #     'direction': 'GZ_TO_ROS'
            # },
            {
                'gz_topic_name': f"/model/{namespace}/pose_static",
                'gz_type_name': 'gz.msgs.Pose_V',
                'ros_topic_name': '/tf_static',
                'ros_type_name': 'tf2_msgs/msg/TFMessage',
                'direction': 'GZ_TO_ROS'
            },
            {
                'gz_topic_name': f'/model/{namespace}/scan',
                'gz_type_name': 'gz.msgs.LaserScan',
                'ros_topic_name': 'scan',
                'ros_type_name': 'sensor_msgs/msg/LaserScan',
                'direction': 'GZ_TO_ROS'
            },
            {
                'gz_topic_name': f"/model/{namespace}/joint_state",
                'gz_type_name': 'gz.msgs.Model',
                'ros_topic_name': 'joint_states',
                'ros_type_name': 'sensor_msgs/msg/JointState',
                'direction': 'GZ_TO_ROS'
            },
            {
                'gz_topic_name': f"/model/{namespace}/tf",
                'gz_type_name': 'gz.msgs.Pose_V',
                'ros_topic_name': '/tf',
                'ros_type_name': 'tf2_msgs/msg/TFMessage',
                'direction': 'GZ_TO_ROS'
            }
    ]

    tmp_cfg = os.path.join(tempfile.gettempdir(), f'bridge_{namespace}.yaml')
    with open(tmp_cfg, 'w') as f:
        yaml.safe_dump(bridge_cfg, f)

    set_namespace = PushRosNamespace(namespace)

    robot_state_publisher = Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            name='robot_state_publisher', output='screen',
            parameters=[ # type: ignore
                {'use_sim_time': True},
                {'expand_gz_topic_names': True },
                {'robot_description': robot_description},
                {'frame_prefix': f'{namespace}/'},
                {'publish_frequency': 30.0}
            ])

    spawn_robot =  Node(
            package='ros_gz_sim', executable='create',
            name='spawn_robot', output='screen',
            arguments=['-topic', 'robot_description', '-name', namespace, '-x', x_offset, '-y', y_offset, '-z', '0.1']
        )

    ros_gz_bridge =  Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name=f'bridge_{namespace}',
            output='screen',
            # these become rclcpp parameters on the bridge node:
            parameters=[
            { 'config_file': tmp_cfg },          # load your YAML topic-map
            { 'use_sim_time': True },                  # standard ROS sim-time flag
            { 'expand_gz_topic_names': True },
            { 'qos_overrides./tf_static.publisher.durability': 'transient_local'}
            ],
        )

    teleop = Node(
            package='teleop_twist_keyboard',
            executable='teleop_twist_keyboard',
            name=f'{namespace}_teleop_twist_keyboard',
            output='screen',
            prefix=f"xterm -T '{namespace} teleop' -e",
            remappings=[('/cmd_vel', 'cmd_vel')]
        )

    world_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=f"world_to_{namespace}_odom",
        arguments=[
            "--x" , x_offset,
            "--y" , y_offset,
            "--z" , "0.0",
            "--roll" , "0.0",
            "--pitch" , "0.0",
            "--yaw" , "0.0",
            "--frame-id", "world",
            "--child-frame-id", f"{namespace}/odom"
        ],
        output="screen",
    )

    return [
        set_namespace,
        robot_state_publisher,
        spawn_robot,
        ros_gz_bridge,
        teleop,
        world_tf
    ]


def generate_launch_description():
    namespace_arg = DeclareLaunchArgument(
        'namespace', default_value='ground_robot',
        description='Namespace and Gazebo entity name'
    )

    x_offset_arg = DeclareLaunchArgument(
        'x', default_value='0.0',
        description='X offset for the robot spawn'
    )

    y_offset_arg = DeclareLaunchArgument(
        'y', default_value='0.0',
        description='Y offset for the robot spawn'
    )

    return LaunchDescription([
        namespace_arg,
        x_offset_arg,
        y_offset_arg,
        OpaqueFunction(function=create_robot_pipeline)
    ])
