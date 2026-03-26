from pathlib import Path
import sys

import cv2
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Tuodaan teidän nykyisestä tiedostosta tunnistus + piirto
from detect_gate import detect_gate, draw_result


class GateBagViewer(Node):
    def __init__(self):
        super().__init__("gate_bag_viewer")

        self.bridge = CvBridge()

        # Vaihda tämä vain jos topicin nimi muuttuu
        self.image_topic = "/image_raw"

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            image_qos
        )

        self.frame_count = 0
        self.get_logger().info(f"Listening to topic: {self.image_topic}")

    def image_callback(self, msg: Image):
        """
        Tämä callback ajetaan aina kun uusi kuva tulee /image_raw-topicista.
        """
        try:
            # ROS-kuvaviesti -> OpenCV-kuva
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Ajetaan teidän nykyinen portin tunnistus
        detection = detect_gate(frame)

        # Piirretään tunnistuksen tulos kuvan päälle
        annotated = draw_result(frame, detection)

        # Näytetään live-ikkunassa
        cv2.imshow("Gate Detection Live From Rosbag", annotated)

        # q = lopeta ikkuna
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.get_logger().info("Q pressed, shutting down.")
            cv2.destroyAllWindows()
            rclpy.shutdown()

        self.frame_count += 1


def main():
    rclpy.init()
    node = GateBagViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()