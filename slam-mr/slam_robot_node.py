import rclpy
import numpy as np
import numpy.typing as npt

from rclpy.subscription import Subscription
from rclpy.node import Node
from typing import Optional, cast
from std_msgs.msg import String
from slam_mr_msgs.msg import BeginRendezvous, Pose
from threading import Lock

def rot_mat(theta: float) -> npt.NDArray[np.float64]:
    """
    Returns a 2D rotation matrix for a given angle in radians.
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def unrot_mat(mat: npt.NDArray[np.float64]) -> float:
    """
    Returns the angle of a 2D rotation matrix.
    """
    assert mat.shape == (2, 2), "Matrix must be 2x2"
    return np.arctan2(mat[1, 0], mat[0, 0])


class SlamRobotNode(Node):

    def __init__(self):
        super().__init__('slam_robot_node')

        # Unique Identifier for the robot.
        self.robot_name = self.get_namespace().split('/')[-1]
        # The robots that are actively rendezvousing with this one alongside their pose estimates.
        self.currently_rendezvousing = cast(Optional[dict[str, Optional[Pose]]], None)
        # The lock to ensure that only one rendezvous is active at a time.
        self.rendezvous_lock = Lock()
        # The ground truth pose estimates (value) of a robot (key) from the perspective of the world.
        self.ground_truth_poses: dict[str, Pose] = {}
        # The pose estimates (value) of the other robots from the perspective of this robot (key).
        self.sensed_pose_estimates: dict[str, Pose] = {}
        # The pose estimates (value) of this robot from the perspective of the other robots (key).
        self.received_pose_estimates: dict[str, Pose] = {}

        # All of the robots that are active
        self.team = {self.robot_name}
        # Adds new robots to the team when they are discovered
        self.team_subscriber = self.create_subscription(String, '/team', self.on_team_msg, 10) # type: ignore
        # Broadcasts the robot name so that it can be discovered by others
        self.team_publisher = self.create_publisher(String, '/team', 10) # type: ignore
        # Publish the robot name every second
        self.team_timer = self.create_timer(1.0, lambda: self.team_publisher.publish(String(data=self.robot_name))) # type: ignore

        # Subscribe to the ground truth pose of all the robots in the team
        self.ground_truth_pose_subscribers: dict[str, Subscription] = {self.robot_name: self.create_subscription(Pose, f'/{self.robot_name}/pose', lambda pose: self.on_ground_truth_pose(pose), 10)} # type: ignore
        # Subscribe to the local pose estimates of the other robots
        self.local_pose_estimate_subscribers: dict[str, Subscription] = {}
        # Broadcasts the local pose estimate of this robot
        self.local_pose_estimate_publisher = self.create_publisher(Pose, f'/{self.robot_name}/local_pose_estimate', 10) # type: ignore
        # Publish the local pose estimate every 1ms
        self.local_pose_timer = self.create_timer(1e-3, f'/{self.robot_name}/local_pose_estimate', self.send_local_pose_estimates) # type: ignore

        # Subscribe to all rendezvous notifications
        self.rendezvous_subscriber = self.create_subscription(BeginRendezvous, '/rendezvous', self.rendezvous_callback, 10) # type: ignore
        # Publish the rendezvous notification
        self.rendezvous_publisher = self.create_publisher(BeginRendezvous, '/rendezvous', 10) # type: ignore
        # Every 30s attempt a rendezvous with the other robots
        self.rendezvous_timer = self.create_timer(30.0, self.start_rendezvous) # type: ignore

        self.get_logger().info(f'Robot name: {self.robot_name}') # type: ignore

    def on_team_msg(self, msg: String):
        """
        Callback for when a new robot is discovered.
        If the robot is not already in the team, add it to the team and subscribe to its ground truth pose and local pose estimates.
        """
        robot_name = cast(str, msg.data) # type: ignore
        if robot_name not in self.team:
            self.team.add(robot_name)
            self.ground_truth_pose_subscribers[robot_name] = \
                self.create_subscription(Pose, f'/{robot_name}/pose', lambda pose: self.on_ground_truth_pose(pose), 10) # type: ignore
            self.local_pose_estimate_subscribers[robot_name] = \
                self.create_subscription(Pose, f'/{robot_name}/local_pose_estimate', lambda pose: self.on_local_pose_estimate(robot_name, pose), 10) # type: ignore

    def on_ground_truth_pose(self, pose: Pose):
        """
        Callback for when a new ground truth pose is received.

        TODO: this function is not done correctly, the pose topic publishes a different msg type and so this would need to be fixed so that the output is indeed a slam_mr_msgs/msg/Pose
        """
        robot_name = cast(str, pose.robot) # type: ignore
        assert robot_name in self.team, f'{self.robot_name} received a ground truth pose from a robot that is not in the team'
        self.ground_truth_poses[robot_name] = pose

        raise NotImplementedError

    def on_local_pose_estimate(self, robot_name: str, pose: Pose):
        """
        Callback for when a new local pose estimate is received.

        If the new message on the topic is a new pose estimate for this robot, update the local pose estimate.
        """
        estimated_robot = cast(str, pose.robot) # type: ignore
        if estimated_robot == self.robot_name:
            self.received_pose_estimates[robot_name] = pose

    def send_local_pose_estimates(self):
        """
        Callback for sending all of the local pose estimates to the other robots in the team.

        This function is called every 1ms and sends the local pose estimates of this robot to all of the other robots in the team.
        It does not send the local pose estimate of this robot nor does it send the local pose estimate of the robots that it is currently rendezvousing with
        Instead, it is assumed that these poses will be sent as they are computed by the rendezvous code.
        """
        for robot_name in self.team:
           if robot_name == self.robot_name or (self.currently_rendezvousing is not None and robot_name in self.currently_rendezvousing):
                continue
           pose = self.get_relative_pose_estimate(robot_name)
           self.sensed_pose_estimates[robot_name] = pose
           self.local_pose_estimate_publisher.publish(pose)

    def get_relative_pose_estimate(self, robot_name: str, w_R: float = 1000, translation_variance: float = .05) -> Pose:
        """
        Computes the noisy relative pose estimate of a robot with respect to this robot.
        This function uses the latest ground truth pose estimates of the given robot and this robot to compute the relative pose estimate.
        """
        rotation_noise = np.random.vonmises(0, w_R)
        translation_noise = np.random.normal(0, translation_variance, (2, 1))

        gt_rotation_i = rot_mat(self.ground_truth_poses[self.robot_name].rotation) # type: ignore
        gt_rotation_j = rot_mat(self.ground_truth_poses[robot_name].rotation) # type: ignore

        gt_translation_i = np.array(self.ground_truth_poses[self.robot_name].translation) # type: ignore
        gt_translation_j = np.array(self.ground_truth_poses[robot_name].translation) # type: ignore

        pose = Pose()
        pose.robot = robot_name
        pose.relative_to = self.robot_name
        pose.rotation = unrot_mat(gt_rotation_i.T @ gt_rotation_j @ rot_mat(rotation_noise)) # type: ignore
        pose.translation = gt_rotation_i.T @ (gt_translation_j - gt_translation_i) + translation_noise # type: ignore

        return pose

    def neighbors(self) -> list[str]:
        """
        Returns the list of neighbors of this robot.
        This function is used to find the robots that are within 0.5 meters of this robot using ground truth poses.
        """
        neighbors: list[str] = []
        for robot_name, pose in self.ground_truth_poses.items():
            if robot_name == self.robot_name:
                continue
            distance = np.linalg.norm(np.array(pose.translation) - np.array(self.ground_truth_poses[self.robot_name].translation)) # type: ignore
            if distance < 0.5:
                neighbors.append(robot_name)
        return neighbors

    def start_rendezvous(self):
        with self.rendezvous_lock:
            if self.currently_rendezvousing is not None:
                return

        self.rendezvous_publisher.publish(BeginRendezvous(initiator=self.robot_name, robots=self.neighbors()))

    def rendezvous_callback(self, msg: BeginRendezvous):
        """
        Listen on /rendezvous for a message to start rendezvousing
        """

        with self.rendezvous_lock:
            if self.currently_rendezvousing is None:
                rendezvousing_robots = cast(list[str], [msg.initiator] + msg.robots) # type: ignore
                if self.robot_name in rendezvousing_robots: # TODO: Check if the robot is contained in the start rendezvous message, make sure to add ego robot
                    other_robots = [robot for robot in rendezvousing_robots if robot != self.robot_name]
                    self.currently_rendezvousing = {robot: None for robot in other_robots}

                    self.get_logger().info(f'{self.robot_name} is rendezvousing with {other_robots}') # type: ignore
                    self.rendezvous(other_robots)

    def rendezvous(self, robot_names: list[str]):
        """
        Performs the necessary steps during rendezvous with the other robots.
        """

        relative_pose_estimates = {self.get_relative_pose_estimate(robot_name) for robot_name in robot_names}
        received_pose_estimates = {robot_name: ... for robot_name in robot_names}
        # From here on out we hsould be sending the new estimates on /robot1/local_pose_estimate


def main(args: Optional[list[str]] = None):
    rclpy.init(args=args)

    slam_robot_node = SlamRobotNode()
    rclpy.spin(slam_robot_node)

    # Destroy the node explicitly
    slam_robot_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()