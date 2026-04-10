#!/usr/bin/env python3
"""
Pose Relay — Fake Particle Filter for gym_ros testing
======================================================
Relays ground truth pose from /ego_racecar/odom to
/pf/viz/inferred_pose (PoseStamped), simulating a
particle filter for testing localized RL policies.

Usage:
    # Terminal 1: ros2 launch f1tenth_gym_ros gym_bridge_launch.py
    # Terminal 2: python3 pose_relay.py
    # Terminal 3: python3 inference_node.py --ros-args ...
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class PoseRelay(Node):
    def __init__(self):
        super().__init__('pose_relay')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('pose_topic', '/pf/viz/inferred_pose')

        odom_topic = self.get_parameter('odom_topic').value
        pose_topic = self.get_parameter('pose_topic').value

        self.sub = self.create_subscription(Odometry, odom_topic, self.cb, 10)
        self.pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.get_logger().info(f'Relaying {odom_topic} → {pose_topic}')

    def cb(self, msg):
        ps = PoseStamped()
        ps.header = msg.header
        ps.header.frame_id = 'map'
        ps.pose = msg.pose.pose
        self.pub.publish(ps)


def main():
    rclpy.init()
    node = PoseRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
