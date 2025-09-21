#!/usr/bin/env python3
"""Replay RTAB-related topics from a rosbag2 recording for offline training.

This node deserializes messages from a rosbag2 file and republishes them with
approximate original timing so the PPO manager can train offline. The replay can
loop and allows optional rate scaling for faster/slower playback.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message

from rosidl_runtime_py.utilities import get_message

try:
    import rosbag2_py
except ImportError as exc:  # pragma: no cover
    rosbag2_py = None
    _ROS2BAG_IMPORT_ERROR = exc
else:
    _ROS2BAG_IMPORT_ERROR = None


class BagReplayNode(Node):
    def __init__(self) -> None:
        super().__init__('rtab_bag_replay')

        if rosbag2_py is None:
            raise RuntimeError(f'rosbag2_py import failed: {_ROS2BAG_IMPORT_ERROR}')

        self.declare_parameter('bag_path', '')
        self.declare_parameter('rate_scale', 1.0)
        self.declare_parameter('loop', False)
        self.declare_parameter('start_offset_sec', 0.0)
        self.declare_parameter('topics', [])  # optional whitelist

        self.bag_path = str(self.get_parameter('bag_path').value)
        if not self.bag_path:
            raise ValueError('bag_path parameter must be set (folder containing metadata.yaml)')

        self.rate_scale = max(float(self.get_parameter('rate_scale').value), 1e-6)
        self.loop = bool(self.get_parameter('loop').value)
        self.start_offset = max(float(self.get_parameter('start_offset_sec').value), 0.0)
        topics_param = self.get_parameter('topics').value
        self.topic_filter = set(topics_param) if topics_param else None

        self.publishers: Dict[str, rclpy.publisher.Publisher] = {}
        self.topic_types: Dict[str, str] = {}

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f'Replay started from {self.bag_path} rate_scale={self.rate_scale} loop={self.loop}'
        )

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        return super().destroy_node()

    # ------------------------------------------------------------------
    def _prepare_publishers(self, metadata: rosbag2_py.Metadata) -> None:
        for topic in metadata.topics_with_message_count:
            name = topic.topic_metadata.name
            type_name = topic.topic_metadata.type
            if self.topic_filter and name not in self.topic_filter:
                continue
            if name not in self.publishers:
                msg_type = get_message(type_name)
                self.publishers[name] = self.create_publisher(msg_type, name, 10)
                self.topic_types[name] = type_name
                self.get_logger().info(f'Publishing replay topic {name} [{type_name}]')

    def _replay_once(self) -> None:
        info_reader = rosbag2_py.Info()
        metadata = info_reader.read_metadata(self.bag_path, 'sqlite3')
        self._prepare_publishers(metadata)

        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        first_time: Optional[int] = None
        start_time = None

        while not self._stop_event.is_set() and reader.has_next():
            topic_name, data, t = reader.read_next()
            if self.topic_filter and topic_name not in self.topic_filter:
                continue
            if topic_name not in self.publishers:
                continue
            if first_time is None:
                first_time = t
                if self.start_offset > 0.0:
                    target_time = first_time + int(self.start_offset * 1e9)
                    while reader.has_next() and t < target_time:
                        topic_name, data, t = reader.read_next()
                        if self.topic_filter and topic_name not in self.topic_filter:
                            continue
                        if topic_name not in self.publishers:
                            continue
                start_time = time.perf_counter()
                first_time = t

            assert first_time is not None and start_time is not None
            elapsed_bag = (t - first_time) / 1e9 / self.rate_scale
            elapsed_real = time.perf_counter() - start_time
            delay = elapsed_bag - elapsed_real
            if delay > 0:
                time.sleep(delay)

            msg_type = get_message(self.topic_types[topic_name])
            msg = deserialize_message(data, msg_type)
            self.publishers[topic_name].publish(msg)

    def _replay_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._replay_once()
                if not self.loop:
                    break
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f'Bag replay error: {exc}')
        finally:
            self.get_logger().info('Replay finished')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BagReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
