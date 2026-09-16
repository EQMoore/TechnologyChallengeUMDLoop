import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

OBSTACLE_DISTANCE_THRESHOLD = 0.5  # meters


class ScanWatcher(Node):

    def __init__(self):
        super().__init__('scan_watcher')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10,
        )

    def scan_callback(self, msg: LaserScan):
        finite_ranges = [r for r in msg.ranges if math.isfinite(r)]
        if not finite_ranges:
            self.get_logger().warn('Scan contained no valid ranges')
            return

        closest = min(finite_ranges)
        if closest < OBSTACLE_DISTANCE_THRESHOLD:
            self.get_logger().warn(f'Obstacle {closest:.2f} m away')
        else:
            self.get_logger().info(f'Closest obstacle: {closest:.2f} m')


def main(args=None):
    rclpy.init(args=args)
    node = ScanWatcher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
