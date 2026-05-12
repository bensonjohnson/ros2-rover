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
            - bev: (legacy) np.array (N, 2, 128, 128) - omit for v3
            - proprio: np.array (N, 6) float32
            - actions: np.array (N, 2) float32
            - rewards: np.array (N, K) float32
            - dones: np.array (N,) bool
            - is_first: np.array (N,) bool
            - rgb: (legacy/optional) np.array (N, 3, 84, 84) uint8 - omit for v3
            - depth: (v3, optional) np.array (N, 1, 72, 96) uint8
            - lidar: (v3, optional) np.array (N, 360) float32
            - is_intervention: (RLPD, optional) np.array (N,) bool
            - is_demo: (RLPD, optional) np.array (N,) bool
            - schema_version: (optional) int. 1=Dreamer, 2=RLPD-legacy, 3=RLPD-v3.
            - metadata: (optional) dict.

    Returns:
        Serialized bytes ready for transport
    """
    compressor = zstd.ZstdCompressor(level=3)

    rewards = np.asarray(batch["rewards"], dtype=np.float32)
    if rewards.ndim == 1:
        rewards = rewards[:, None]
    n_steps = rewards.shape[0]

    compressed = {
        # Small arrays don't benefit from compression
        "proprio": batch["proprio"].tolist(),
        "actions": batch["actions"].tolist(),
        # rewards is (N, K); .tolist() yields a list of K-length lists
        "rewards": rewards.tolist(),
        "reward_channels": int(rewards.shape[1]),
        "dones": batch["dones"].tolist(),
        "is_first": batch.get("is_first", np.zeros(n_steps, dtype=bool)).tolist(),
        "schema_version": int(batch.get("schema_version", 1)),
        "metadata": batch.get("metadata", {}),
    }

    # Legacy BEV (Dreamer + RLPD v2). v3 chunks omit it.
    if "bev" in batch and batch["bev"] is not None:
        compressed["bev"] = {
            "data": compressor.compress(batch["bev"].tobytes()),
            "shape": batch["bev"].shape,
            "dtype": str(batch["bev"].dtype),
        }

    # Legacy RGB stream (Dreamer + RLPD v2). v3 chunks omit it.
    if "rgb" in batch and batch["rgb"] is not None:
        compressed["rgb"] = {
            "data": compressor.compress(batch["rgb"].tobytes()),
            "shape": batch["rgb"].shape,
            "dtype": str(batch["rgb"].dtype),
        }

    # v3 depth stream (uint8 96x72, zstd compresses ~10x)
    if "depth" in batch and batch["depth"] is not None:
        compressed["depth"] = {
            "data": compressor.compress(batch["depth"].tobytes()),
            "shape": batch["depth"].shape,
            "dtype": str(batch["depth"].dtype),
        }

    # v3 lidar stream (float32 360 beams, compresses ~3x via spatial correlation)
    if "lidar" in batch and batch["lidar"] is not None:
        compressed["lidar"] = {
            "data": compressor.compress(batch["lidar"].tobytes()),
            "shape": batch["lidar"].shape,
            "dtype": str(batch["lidar"].dtype),
        }

    # RLPD / HIL-SERL flags (optional). Only emitted if the rover provided them,
    # so Dreamer chunks stay byte-identical on the wire.
    if "is_intervention" in batch:
        compressed["is_intervention"] = np.asarray(
            batch["is_intervention"], dtype=bool
        ).tolist()
    if "is_demo" in batch:
        compressed["is_demo"] = np.asarray(batch["is_demo"], dtype=bool).tolist()

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
        "proprio": np.array(compressed["proprio"], dtype=np.float32),
        "actions": np.array(compressed["actions"], dtype=np.float32),
        "rewards": rewards,
        "reward_channels": int(compressed.get("reward_channels", rewards.shape[1])),
        "dones": np.array(compressed["dones"], dtype=bool),
        "is_first": np.array(compressed.get("is_first", [False] * n_steps), dtype=bool),
        "schema_version": int(compressed.get("schema_version", 1)),
        "metadata": compressed.get("metadata", {}),
    }

    # Legacy BEV (Dreamer + RLPD v2)
    if compressed.get("bev") is not None:
        result["bev"] = np.frombuffer(
            decompressor.decompress(compressed["bev"]["data"]),
            dtype=np.dtype(compressed["bev"]["dtype"])
        ).reshape(compressed["bev"]["shape"])

    # Legacy RGB (Dreamer + RLPD v2)
    if compressed.get("rgb") is not None:
        result["rgb"] = np.frombuffer(
            decompressor.decompress(compressed["rgb"]["data"]),
            dtype=np.dtype(compressed["rgb"]["dtype"])
        ).reshape(compressed["rgb"]["shape"])

    # v3 depth + lidar
    if compressed.get("depth") is not None:
        result["depth"] = np.frombuffer(
            decompressor.decompress(compressed["depth"]["data"]),
            dtype=np.dtype(compressed["depth"]["dtype"])
        ).reshape(compressed["depth"]["shape"])
    if compressed.get("lidar") is not None:
        result["lidar"] = np.frombuffer(
            decompressor.decompress(compressed["lidar"]["data"]),
            dtype=np.dtype(compressed["lidar"]["dtype"])
        ).reshape(compressed["lidar"]["shape"])

    # RLPD / HIL-SERL flags. Absent on Dreamer chunks → caller gets all-False.
    if compressed.get("is_intervention") is not None:
        result["is_intervention"] = np.array(
            compressed["is_intervention"], dtype=bool
        )
    if compressed.get("is_demo") is not None:
        result["is_demo"] = np.array(compressed["is_demo"], dtype=bool)

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
