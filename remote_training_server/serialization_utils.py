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
            - laser: np.array of shape (N, 128, 128) float32
            - depth: np.array of shape (N, 424, 240) float32
            - proprio: np.array of shape (N, 10) float32
            - actions: np.array of shape (N, 2) float32
            - rewards: np.array of shape (N,) float32
            - dones: np.array of shape (N,) bool

    Returns:
        Serialized bytes ready for NATS publish
    """
    compressor = zstd.ZstdCompressor(level=3)

    # Compress large arrays
    compressed = {
        "laser": {
            "data": compressor.compress(batch["laser"].tobytes()),
            "shape": batch["laser"].shape,
            "dtype": str(batch["laser"].dtype),
        },
        "depth": {
            "data": compressor.compress(batch["depth"].tobytes()),
            "shape": batch["depth"].shape,
            "dtype": str(batch["depth"].dtype),
        },
        # Small arrays don't benefit from compression
        "proprio": batch["proprio"].tolist(),
        "actions": batch["actions"].tolist(),
        "rewards": batch["rewards"].tolist(),
        "dones": batch["dones"].tolist(),
    }

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

    return {
        "laser": np.frombuffer(
            decompressor.decompress(compressed["laser"]["data"]),
            dtype=np.dtype(compressed["laser"]["dtype"])
        ).reshape(compressed["laser"]["shape"]),
        "depth": np.frombuffer(
            decompressor.decompress(compressed["depth"]["data"]),
            dtype=np.dtype(compressed["depth"]["dtype"])
        ).reshape(compressed["depth"]["shape"]),
        "proprio": np.array(compressed["proprio"], dtype=np.float32),
        "actions": np.array(compressed["actions"], dtype=np.float32),
        "rewards": np.array(compressed["rewards"], dtype=np.float32),
        "dones": np.array(compressed["dones"], dtype=bool),
    }


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
