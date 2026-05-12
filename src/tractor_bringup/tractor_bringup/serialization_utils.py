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
            - proprio: np.array of shape (N, 12) float32
            - actions: np.array of shape (N, 2) float32
            - rewards: np.array of shape (N,) float32 OR (N, K) float32 — K reward channels
            - dones: np.array of shape (N,) bool
            - is_intervention: (optional, RLPD) np.array of shape (N,) bool —
              True for steps where the human took over via deadman during autonomy.
            - is_demo: (optional, RLPD) np.array of shape (N,) bool —
              True for entire teleop collection chunks.
            - schema_version: (optional) int. Default 1 (PPO/Dreamer). RLPD ships 2.
            - metadata: (optional) dict with rover_id, model_id, etc.

    Returns:
        Serialized bytes ready for NATS publish
    """
    compressor = zstd.ZstdCompressor(level=3)

    # Convert is_eval to list if it exists (backward compatibility handled in extraction)
    is_eval_data = batch.get("is_eval", np.zeros(len(batch["rewards"]), dtype=bool)).tolist()

    rewards = np.asarray(batch["rewards"], dtype=np.float32)
    if rewards.ndim == 1:
        rewards = rewards[:, None]
    n_steps = rewards.shape[0]

    compressed = {
        # Small arrays don't benefit from compression
        "proprio": batch["proprio"].tolist(),
        "actions": batch["actions"].tolist(),
        "rewards": rewards.tolist(),
        "reward_channels": int(rewards.shape[1]),
        "dones": batch["dones"].tolist(),
        "is_eval": is_eval_data,
        # is_first flag per step (used by DreamerV3 to reset RSSM state on episode boundaries).
        # Optional — old PPO rollouts without this key continue to work.
        "is_first": batch.get("is_first", np.zeros(n_steps, dtype=bool)).tolist(),
        "schema_version": int(batch.get("schema_version", 1)),
        # Include metadata if present
        "metadata": batch.get("metadata", {}),
    }

    # Legacy BEV (Dreamer / RLPD v2). v3 omits it.
    if "bev" in batch and batch["bev"] is not None:
        compressed["bev"] = {
            "data": compressor.compress(batch["bev"].tobytes()),
            "shape": batch["bev"].shape,
            "dtype": str(batch["bev"].dtype),
        }

    # Legacy RGB (Dreamer / RLPD v2). v3 omits it.
    if "rgb" in batch and batch["rgb"] is not None:
        compressed["rgb"] = {
            "data": compressor.compress(batch["rgb"].tobytes()),
            "shape": batch["rgb"].shape,
            "dtype": str(batch["rgb"].dtype),
        }

    # v3 depth stream (uint8 96×72, zstd-friendly)
    if "depth" in batch and batch["depth"] is not None:
        compressed["depth"] = {
            "data": compressor.compress(batch["depth"].tobytes()),
            "shape": batch["depth"].shape,
            "dtype": str(batch["depth"].dtype),
        }

    # v3 lidar stream (float32 360 beams)
    if "lidar" in batch and batch["lidar"] is not None:
        compressed["lidar"] = {
            "data": compressor.compress(batch["lidar"].tobytes()),
            "shape": batch["lidar"].shape,
            "dtype": str(batch["lidar"].dtype),
        }

    # RLPD / HIL-SERL flags (optional). Only emitted if the rover provided them,
    # so PPO/Dreamer chunks stay byte-identical on the wire.
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
        data: Serialized bytes from NATS message

    Returns:
        Dictionary with numpy arrays reconstructed
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
        "is_eval": np.array(compressed.get("is_eval", [False] * n_steps), dtype=bool),
        "is_first": np.array(compressed.get("is_first", [False] * n_steps), dtype=bool),
        "schema_version": int(compressed.get("schema_version", 1)),
        "metadata": compressed.get("metadata", {}),
    }

    # Legacy BEV (Dreamer / RLPD v2)
    if compressed.get("bev") is not None:
        result["bev"] = np.frombuffer(
            decompressor.decompress(compressed["bev"]["data"]),
            dtype=np.dtype(compressed["bev"]["dtype"])
        ).reshape(compressed["bev"]["shape"])

    # Legacy RGB (Dreamer / RLPD v2)
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

    # RLPD / HIL-SERL flags. Absent on Dreamer/PPO chunks → caller doesn't see them.
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
