"""
Serialization utilities for NATS JetStream communication.
Provides efficient serialization/deserialization of training data using
MessagePack and Zstandard compression.
"""

import numpy as np
import msgpack
import msgpack_numpy as m
import zstandard as zstd

# Enable NumPy support in msgpack
m.patch()


def serialize_batch(batch: dict) -> bytes:
    """
    Serialize an experience batch with compression.

    Args:
        batch: Dictionary containing:
            - bev: np.array of shape (N, 2, 128, 128) float32 - Unified BEV grid
            - proprio: np.array of shape (N, 6) float32
            - actions: np.array of shape (N, 2) float32
            - rewards: np.array of shape (N, K) float32 — K reward channels
            - dones: np.array of shape (N,) bool
            - is_first: np.array of shape (N,) bool
            - rgb: (optional) np.array of shape (N, 3, 84, 84) uint8/float32
            - metadata: (optional) dict with rover_id, model_id, etc.

    Returns:
        Serialized bytes ready for transport
    """
    compressor = zstd.ZstdCompressor(level=3)

    rewards = np.asarray(batch["rewards"], dtype=np.float32)
    if rewards.ndim == 1:
        rewards = rewards[:, None]
    n_steps = rewards.shape[0]

    compressed = {
        "bev": {
            "data": compressor.compress(batch["bev"].tobytes()),
            "shape": batch["bev"].shape,
            "dtype": str(batch["bev"].dtype),
        },
        # Small arrays don't benefit from compression
        "proprio": batch["proprio"].tolist(),
        "actions": batch["actions"].tolist(),
        # rewards is (N, K); .tolist() yields a list of K-length lists
        "rewards": rewards.tolist(),
        "reward_channels": int(rewards.shape[1]),
        "dones": batch["dones"].tolist(),
        "is_first": batch.get("is_first", np.zeros(n_steps, dtype=bool)).tolist(),
        "metadata": batch.get("metadata", {}),
    }

    # RGB stream (optional, compressed like BEV)
    if "rgb" in batch:
        compressed["rgb"] = {
            "data": compressor.compress(batch["rgb"].tobytes()),
            "shape": batch["rgb"].shape,
            "dtype": str(batch["rgb"].dtype),
        }

    return msgpack.packb(compressed)


def deserialize_batch(data: bytes) -> dict:
    """
    Deserialize an experience batch.

    Args:
        data: Serialized bytes from transport

    Returns:
        Dictionary with numpy arrays reconstructed. `rewards` is always (N, K)
        where K is the number of reward channels.
    """
    decompressor = zstd.ZstdDecompressor()
    compressed = msgpack.unpackb(data)

    rewards = np.array(compressed["rewards"], dtype=np.float32)
    if rewards.ndim == 1:
        rewards = rewards[:, None]
    n_steps = rewards.shape[0]

    result = {
        "bev": np.frombuffer(
            decompressor.decompress(compressed["bev"]["data"]),
            dtype=np.dtype(compressed["bev"]["dtype"])
        ).reshape(compressed["bev"]["shape"]),
        "proprio": np.array(compressed["proprio"], dtype=np.float32),
        "actions": np.array(compressed["actions"], dtype=np.float32),
        "rewards": rewards,
        "reward_channels": int(compressed.get("reward_channels", rewards.shape[1])),
        "dones": np.array(compressed["dones"], dtype=bool),
        "is_first": np.array(compressed.get("is_first", [False] * n_steps), dtype=bool),
        "metadata": compressed.get("metadata", {}),
    }

    # RGB stream (optional)
    if compressed.get("rgb") is not None:
        result["rgb"] = np.frombuffer(
            decompressor.decompress(compressed["rgb"]["data"]),
            dtype=np.dtype(compressed["rgb"]["dtype"])
        ).reshape(compressed["rgb"]["shape"])

    return result


def serialize_model_update(onnx_bytes: bytes, version: int) -> bytes:
    """
    Serialize a model update message.

    Args:
        onnx_bytes: Raw ONNX model file bytes
        version: Model version number

    Returns:
        Serialized message
    """
    return msgpack.packb({
        "onnx_bytes": onnx_bytes,
        "version": version,
    })


def deserialize_model_update(data: bytes) -> dict:
    """
    Deserialize a model update message.

    Args:
        data: Serialized bytes from NATS

    Returns:
        Dictionary with 'onnx_bytes' and 'version'
    """
    return msgpack.unpackb(data)



def serialize_metadata(version: int, timestamp: float) -> bytes:
    """
    Serialize model metadata message.

    Args:
        version: Latest model version
        timestamp: Unix timestamp

    Returns:
        Serialized metadata
    """
    return msgpack.packb({
        "latest_version": version,
        "timestamp": timestamp,
    })


def deserialize_metadata(data: bytes) -> dict:
    """
    Deserialize model metadata message.

    Args:
        data: Serialized bytes from NATS

    Returns:
        Dictionary with 'latest_version' and 'timestamp'
    """
    return msgpack.unpackb(data)


def serialize_status(status: str, **kwargs) -> bytes:
    """
    Serialize a status message.

    Args:
        status: Status string ('ready', 'training', etc.)
        **kwargs: Additional status fields

    Returns:
        Serialized status message
    """
    msg = {"status": status}
    msg.update(kwargs)
    return msgpack.packb(msg)


def deserialize_status(data: bytes) -> dict:
    """
    Deserialize a status message.

    Args:
        data: Serialized bytes from NATS

    Returns:
        Status dictionary
    """
    return msgpack.unpackb(data)
