cd turtlebot3_ws/
source /opt/ros/humble/setup.bash
. /usr/share/gazebo/setup.sh
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo empty_world.launch.py
