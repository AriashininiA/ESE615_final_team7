from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path = LaunchConfiguration("model_path")
    config_path = LaunchConfiguration("config_path")
    max_speed = LaunchConfiguration("max_speed")
    scan_topic = LaunchConfiguration("scan_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    drive_topic = LaunchConfiguration("drive_topic")
    use_onnx = LaunchConfiguration("use_onnx")
    inference_rate = LaunchConfiguration("inference_rate")
    smoothing_alpha = LaunchConfiguration("smoothing_alpha")
    max_steer_rate = LaunchConfiguration("max_steer_rate")
    flip_scan = LaunchConfiguration("flip_scan")
    foxglove_port = LaunchConfiguration("foxglove_port")
    start_foxglove = LaunchConfiguration("start_foxglove")

    return LaunchDescription([
        DeclareLaunchArgument("model_path", default_value="runs/bc_ppo_curriculum_levine_safe_2026-04-27_11-28-43/final_model.zip"),
        DeclareLaunchArgument("config_path", default_value="runs/bc_ppo_curriculum_levine_safe_2026-04-27_11-28-43/config.yaml"),
        DeclareLaunchArgument("max_speed", default_value="2.0"),
        DeclareLaunchArgument("scan_topic", default_value="/scan"),
        DeclareLaunchArgument("odom_topic", default_value="/odom"),
        DeclareLaunchArgument("drive_topic", default_value="/drive"),
        DeclareLaunchArgument("use_onnx", default_value="false"),
        DeclareLaunchArgument("inference_rate", default_value="40.0"),
        DeclareLaunchArgument("smoothing_alpha", default_value="0.4"),
        DeclareLaunchArgument("max_steer_rate", default_value="2.0"),
        DeclareLaunchArgument("flip_scan", default_value="false"),
        DeclareLaunchArgument("start_foxglove", default_value="true"),
        DeclareLaunchArgument("foxglove_port", default_value="8765"),

        Node(
            package="f1tenth_rl_deploy",
            executable="rl_inference",
            name="rl_inference_node",
            output="screen",
            parameters=[{
                "model_path": model_path,
                "config_path": config_path,
                "max_speed": max_speed,
                "scan_topic": scan_topic,
                "odom_topic": odom_topic,
                "drive_topic": drive_topic,
                "use_onnx": use_onnx,
                "inference_rate": inference_rate,
                "smoothing_alpha": smoothing_alpha,
                "max_steer_rate": max_steer_rate,
                "flip_scan": flip_scan,
            }],
        ),

        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            output="screen",
            parameters=[{
                "port": foxglove_port,
                "address": "0.0.0.0",
            }],
            condition=IfCondition(start_foxglove),
        ),
    ])
