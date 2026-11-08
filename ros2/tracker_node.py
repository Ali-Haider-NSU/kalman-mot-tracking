import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D
# the SORT class from this tutorial, saved as a module
from sort_tracker import SORT
import numpy as np


class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker")
        self.declare_parameter("iou_min", 0.3)
        self.declare_parameter("max_age", 30)
        self.tracker = SORT(iou_min=self.get_parameter("iou_min").value,
                            max_age=self.get_parameter("max_age").value)
        self.sub_ = self.create_subscription(Detection2DArray, "detections",
                                             self.on_dets, 10)
        self.pub_ = self.create_publisher(Detection2DArray, "tracks", 10)

    def on_dets(self, msg):
        rows = []
        for d in msg.detections:
            cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
            w, h = d.bbox.size_x, d.bbox.size_y
            score = d.results[0].hypothesis.score if d.results else 1.0
            rows.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score])
        tracks = self.tracker.step(np.array(rows).reshape(-1, 5))

        out = Detection2DArray()
        out.header = msg.header                       # keep the image timestamp and frame
        for tid, b in tracks:
            t = Detection2D()
            # the track ID travels in Detection2D.id
            t.id = str(tid)
            t.bbox.center.position.x = float((b[0] + b[2]) / 2)
            t.bbox.center.position.y = float((b[1] + b[3]) / 2)
            t.bbox.size_x, t.bbox.size_y = float(b[2] - b[0]), float(b[3] - b[1])
            out.detections.append(t)
        self.pub_.publish(out)


def main():
    rclpy.init()
    rclpy.spin(TrackerNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
