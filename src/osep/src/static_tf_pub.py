#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

class StaticTfPublisher(Node):
    def __init__(self):
        super().__init__('static_tf_pub')

        # Create static broadcaster
        self.broadcaster = StaticTransformBroadcaster(self)

        # Define static transform
        static_tf = TransformStamped()
        static_tf.header.stamp = self.get_clock().now().to_msg()
        static_tf.header.frame_id = 'odom'          # parent
        static_tf.child_frame_id = 'base_link'     # child

        # Translation (change these if needed)
        static_tf.transform.translation.x = 0.0
        static_tf.transform.translation.y = 0.0
        static_tf.transform.translation.z = 0.0

        # Rotation (identity quaternion = no rotation)
        static_tf.transform.rotation.x = 0.0
        static_tf.transform.rotation.y = 0.0
        static_tf.transform.rotation.z = 0.0
        static_tf.transform.rotation.w = 1.0

        # Send it
        self.broadcaster.sendTransform(static_tf)
        self.get_logger().info('Published static transform odom → base_link')


def main(args=None):
    rclpy.init(args=args)
    node = StaticTfPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
