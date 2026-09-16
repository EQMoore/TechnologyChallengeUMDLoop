import math
import random

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import tf2_ros
from tf2_ros import TransformException
from laser_geometry import LaserProjection
from geometry_msgs.msg import Twist, TwistStamped

from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import heapq
import matplotlib.pyplot as plt


class Navigation(Node):

    def __init__(self):
        super().__init__('navigation')

        self.callback_group = ReentrantCallbackGroup()
        self.plan_timer = self.create_timer(1.0, self.plan_callback, callback_group=self.callback_group)
        self.control_timer = self.create_timer(0.1, self.control_callback, callback_group=self.callback_group)
        self.current_path = None
        self.point_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10,
        )
        self.occupancy_grid = np.zeros((10, 10), dtype=np.int8)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.projector = LaserProjection()
        self.targets = []
        self.all_targets = []
        self.path_taken = []

        self.cmd_vel_publisher = self.create_publisher(TwistStamped, 'diff_drive_controller/cmd_vel', 10)

        self.grid_scale = 1
        self.grid_offset = 5

        self.waypoint_tolerance = 0.15
        self.goal_tolerance = 0.2
        self.max_linear_speed = 1.0
        self.max_angular_speed = 0.5
        self.k_linear = 0.7
        self.k_angular = 0.2

        self.control_period = 0.1
        self.max_linear_accel = 0.5
        self.max_angular_accel = 2.0 
        self.last_linear_speed = 0.0
        self.last_angular_speed = 0.0

    # A helper method to generate the TwistStamped message (Twist + Timestamp) and publish it to the cmd_vel topic. 
    #It takes a Twist message as input, creates a new TwistStamped message, 
    # sets its header and twist fields, and publishes it to the cmd_vel topic.
    def publish_cmd_vel(self, twist):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist = twist
        self.cmd_vel_publisher.publish(msg)

    # The plan callback runs off of a timer that runs at a higher frequency than the control callback.
    # It checks if there are any targets to reach, and if so it computes a path to the next target using the A* algorithm and stores it in the current_path variable.
    # If there are no targets, it sets the current_path to None. If the robot has reached the current target,
    # it removes it from the list of targets and sets the current_path to None.

    def plan_callback(self):
        if not self.targets:
            self.current_path = None
            return

        try:
            x, y, _ = self.get_robot_pose()
        except TransformException as ex:
            self.get_logger().warn(f"Failed to look up robot pose: {str(ex)}")
            return

        robot_cell = self.world_to_grid(x, y)
        goal_cell = self.targets[0]

        if not self.path_taken or self.path_taken[-1] != robot_cell:
            self.path_taken.append(robot_cell)

        if self.distance((x, y), self.grid_to_world(goal_cell)) < self.goal_tolerance:
            self.get_logger().info(f"Reached target {goal_cell}.")
            self.current_path = None
            self.targets.pop(0)
            return

        path = self.astar(self.occupancy_grid, robot_cell, goal_cell)
        if path:
            self.current_path = path[1:] if len(path) > 1 else path
        else:
            self.get_logger().info("No path found.")
            self.current_path = None
            self.targets.pop(0)

    # The control callback runs off of a timer that runs at a lower frequency from the plan callback
    # It checks if there is a current path to follow, and if so it computes the control command to follow that path and publishes it to the cmd_vel topic. 
    # If there is no current path, it stops the robot by publishing a zero velocity command.

    def control_callback(self):
        if not self.current_path:
            self.last_linear_speed = 0.0
            self.last_angular_speed = 0.0
            self.publish_cmd_vel(Twist())
            return

        try:
            twist = self.compute_control(self.current_path)
        except TransformException as exeption:
            self.get_logger().warn(f"Failed to look up robot pose: {str(exeption)}")
            return
        self.publish_cmd_vel(twist)

    # Turn the polar laser points into a cartesian point cloud, than transfrom through the TF tree down to the odometry frame
    # Then call the update_occupancy_grid function to update the occupancy grid with the new points
    def scan_callback(self, msg):
        try:
            cloud_out = self.projector.projectLaser(msg)
            transform = self.tf_buffer.lookup_transform(
                target_frame='odom',
                source_frame=msg.header.frame_id,
                time=msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            world_cloud = do_transform_cloud(cloud_out, transform)
            self.update_occupancy_grid(world_cloud)
        except TransformException as exeption:
            self.get_logger().warn(f"Failed to transform laser scan: {str(exeption)}")

    #From the world framed point cloud, update the occupancy grid by setting the corresponding cells to 1 (occupied)
    
    def update_occupancy_grid(self, cloud_out):
        points = np.array(list(pc2.read_points(cloud_out, field_names=("x", "y"), skip_nans=True)))
        for point in points:
            x_idx, y_idx = self.world_to_grid(point[0], point[1])
            if 0 <= x_idx < self.occupancy_grid.shape[0] and 0 <= y_idx < self.occupancy_grid.shape[1]:
                self.occupancy_grid[x_idx, y_idx] = 1

    # query robot pose from the tf buffer, which is a transform from the odom frame to the base_footprint frame. 
    # The odom frame is a fixed frame that represents the robot's position in the world,
    # while the base_footprint frame is a moving frame that represents the robot's position 
    # relative to its own body. The transform contains the translation and rotation of the robot in 3D space,
    # which we can use to compute its x, y, and yaw (orientation) values.

    def get_robot_pose(self):
        transform = self.tf_buffer.lookup_transform(
            target_frame='odom',
            source_frame='base_footprint',
            time=rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=0.1),
        )

        x = transform.transform.translation.x
        y = transform.transform.translation.y
        yaw = self.yaw_from_quaternion(transform.transform.rotation)
        return (x, y, yaw)

    def world_to_grid(self, x, y):
        return (int(x * self.grid_scale) + self.grid_offset, int(y * self.grid_scale) + self.grid_offset)

    def grid_to_world(self, cell):
        return ((cell[0] - self.grid_offset) / self.grid_scale, (cell[1] - self.grid_offset) / self.grid_scale)

    #For reference I'm not cracked I have no clue what quaternians do I just know they are an algebra that is an extension of the complex numbers and are used to represent rotations in 3D space.
    @staticmethod
    def yaw_from_quaternion(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
    
    #normalize the angle to be between -pi and pi so that the robot can turn in the shortest direction
    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    #pythagorean theorem to find the distance between two points in a 2D space
    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def compute_control(self, path):
        twist = Twist()

        if not path:
            return twist

        x, y, yaw = self.get_robot_pose()

        # Drop waypoints we've already reached.
        while len(path) > 1 and self.distance((x, y), self.grid_to_world(path[0])) < self.waypoint_tolerance:
            path.pop(0)

        target_x, target_y = self.grid_to_world(path[0])
        distance_to_target = self.distance((x, y), (target_x, target_y))

        if len(path) == 1 and distance_to_target < self.goal_tolerance:
            path.clear()
            self.last_linear_speed = 0.0
            self.last_angular_speed = 0.0
            return twist

        angle_to_target = math.atan2(target_y - y, target_x - x)
        angle_error = self.normalize_angle(angle_to_target - yaw)

        desired_linear = max(0.0, min(self.max_linear_speed, self.k_linear * distance_to_target * math.cos(angle_error)))
        desired_angular = max(-self.max_angular_speed, min(self.max_angular_speed, self.k_angular * angle_error))

        # Ramp toward the desired speeds instead of snapping to them, so a
        # noisy/jumpy angle_error can't fling the robot from full-speed one
        # way to full-speed the other way in a single control tick.
        max_dv = self.max_linear_accel * self.control_period
        max_dw = self.max_angular_accel * self.control_period
        twist.linear.x = self.last_linear_speed + max(-max_dv, min(max_dv, desired_linear - self.last_linear_speed))
        twist.angular.z = self.last_angular_speed + max(-max_dw, min(max_dw, desired_angular - self.last_angular_speed))

        self.last_linear_speed = twist.linear.x
        self.last_angular_speed = twist.angular.z

        return twist

    def astar(self, occupancy_grid, start, goal):
        rows, cols = occupancy_grid.shape
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def manhattan_heuristic(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        open_set = [(manhattan_heuristic(start, goal), start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for dr, dc in neighbors:
                neighbor = (current[0] + dr, current[1] + dc)

                if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                    continue
                if occupancy_grid[neighbor] == 1:
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + manhattan_heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None


def generate_targets(self, targets):
    for _ in range(targets):
        x_idx = random.randint(0, self.occupancy_grid.shape[0] - 1)
        y_idx = random.randint(0, self.occupancy_grid.shape[1] - 1)
        self.targets.append((x_idx, y_idx))
        self.all_targets.append((x_idx, y_idx))


def plot_occupancy_grid(occupancy_grid, path_taken=None, targets=None):
    plt.imshow(occupancy_grid, cmap='gray', origin='lower')
    plt.title('Occupancy Grid')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Occupancy Value')

    if path_taken:
        path_x = [cell[1] for cell in path_taken]
        path_y = [cell[0] for cell in path_taken]
        plt.plot(path_x, path_y, color='red', marker='o', markersize=3, linewidth=1, label='Path Taken')

    if targets:
        target_x = [cell[1] for cell in targets]
        target_y = [cell[0] for cell in targets]
        plt.scatter(target_x, target_y, color='cyan', marker='*', s=150, label='Targets')

    if path_taken or targets:
        plt.legend()

    plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = Navigation()
    generate_targets(node, 3)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        plot_occupancy_grid(node.occupancy_grid, node.path_taken, node.all_targets)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
