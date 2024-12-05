#!/usr/bin/env python3

import rclpy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

# Global publisher
cmd_vel_publisher = None

def gesture_callback(msg):
    global cmd_vel_publisher
    gesture = msg.data.lower().strip()  # Process the gesture message

    # Initialize a Twist message
    twist = Twist()

    if gesture == "forward":
        twist.linear.x = 0.2 
        twist.angular.z = 0.0
        print('Forward')
    elif gesture == "left":
        twist.linear.x = 0.0
        twist.angular.z = 0.5 
        print('Left')
    elif gesture == "right":
        twist.linear.x = 0.0
        twist.angular.z = -0.5  
        print('Right')
    elif gesture == "stop":
        twist.linear.x = 0.0
        twist.angular.z = 0.0 
        print('Stop')
    else:
        return  

    # Publish the Twist message
    cmd_vel_publisher.publish(twist)

def main(args=None):
    global cmd_vel_publisher

    # Initialize the ROS2 node
    rclpy.init(args=args)
    node = rclpy.create_node('gesture_to_twist_node')

    # Publisher for /cmd_vel
    cmd_vel_publisher = node.create_publisher(Twist, '/cmd_vel', 10)

    # Subscriber for /atc/gestures
    node.create_subscription(String, '/atc/orders', gesture_callback, 10)

    print('GestureToTwistNode is running...')

    try:
        # Spin the node
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Shutting down GestureToTwistNode...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
