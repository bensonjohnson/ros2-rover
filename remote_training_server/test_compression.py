#!/usr/bin/env python3
"""Test different compression algorithms for trajectory data."""

import time
import numpy as np
import lz4.frame

# Test if other compression libraries are available
try:
    import lzma
    HAS_LZMA = True
except ImportError:
    HAS_LZMA = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import blosc2
    HAS_BLOSC = True
except ImportError:
    HAS_BLOSC = False


def generate_test_data():
    """Generate synthetic trajectory data similar to real rover data."""
    # Simulate 50 frames at 10Hz = 5 second episode
    num_frames = 50

    # RGB: 50 frames of 240x320x3 (typical camera resolution after resize)
    # Start with structured data (more realistic than pure random)
    rgb = np.zeros((num_frames, 240, 320, 3), dtype=np.uint8)

    # Generate realistic-ish image with spatial structure
    for i in range(num_frames):
        # Sky gradient (top half)
        for y in range(120):
            rgb[i, y, :, 0] = 100 + y // 2  # Red channel gradient
            rgb[i, y, :, 1] = 120 + y // 2  # Green channel gradient
            rgb[i, y, :, 2] = 180 + y // 3  # Blue channel gradient

        # Ground (bottom half) - more variation
        for y in range(120, 240):
            rgb[i, y, :, :] = 80 + (y - 120) // 3

        # Add objects (simulate obstacles)
        rgb[i, 100:140, 50:100, :] = [139, 69, 19]  # Brown obstacle 1
        rgb[i, 80:120, 200:240, :] = [34, 139, 34]  # Green obstacle 2

        # Add noise to simulate camera noise (but much less than pure random)
        noise = np.random.randint(-10, 10, size=rgb[i].shape, dtype=np.int16)
        rgb[i] = np.clip(rgb[i].astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Temporal correlation - slight movement
        if i > 0:
            # 80% same as previous frame (simulates slow rover movement)
            blend_ratio = 0.8
            rgb[i] = (rgb[i-1] * blend_ratio + rgb[i] * (1 - blend_ratio)).astype(np.uint8)

    # Depth: 50 frames of 240x320 (float32)
    depth = np.ones((num_frames, 240, 320), dtype=np.float32) * 3.0  # Default 3m

    # Add depth structure (smoother than RGB)
    for i in range(num_frames):
        # Gradient - closer at bottom, farther at top
        for y in range(240):
            depth[i, y, :] = 5.0 - (y / 240) * 3.0

        # Obstacles at fixed depths
        depth[i, 100:140, 50:100] = 1.2  # Obstacle 1 at 1.2m
        depth[i, 80:120, 200:240] = 1.5  # Obstacle 2 at 1.5m

        # Small amount of noise
        noise = np.random.randn(240, 320) * 0.05
        depth[i] = np.clip(depth[i] + noise, 0.1, 10.0)

        # Temporal correlation
        if i > 0:
            depth[i] = depth[i-1] * 0.85 + depth[i] * 0.15

    return rgb, depth


def test_compression(data, name, compress_fn, decompress_fn, level=None):
    """Test a compression algorithm."""
    data_bytes = data.tobytes()
    original_size = len(data_bytes)

    # Compression
    start = time.time()
    if level is not None:
        compressed = compress_fn(data_bytes, level)
    else:
        compressed = compress_fn(data_bytes)
    compress_time = time.time() - start

    compressed_size = len(compressed)
    ratio = original_size / compressed_size

    # Decompression
    start = time.time()
    decompressed = decompress_fn(compressed)
    decompress_time = time.time() - start

    # Verify
    assert len(decompressed) == original_size, "Decompression failed!"

    return {
        'name': name,
        'original_mb': original_size / 1024 / 1024,
        'compressed_mb': compressed_size / 1024 / 1024,
        'ratio': ratio,
        'compress_time_ms': compress_time * 1000,
        'decompress_time_ms': decompress_time * 1000,
    }


def main():
    print("=" * 80)
    print("Compression Algorithm Comparison for Rover Trajectory Data")
    print("=" * 80)
    print()

    # Generate test data
    print("Generating test data (50 frames, ~22 MB)...")
    rgb, depth = generate_test_data()
    print(f"  RGB shape: {rgb.shape}")
    print(f"  Depth shape: {depth.shape}")
    print()

    results = []

    # Test LZ4 (current)
    print("Testing LZ4 (current)...")
    result = test_compression(
        rgb, "LZ4 (RGB)",
        lz4.frame.compress,
        lz4.frame.decompress
    )
    results.append(result)

    result = test_compression(
        depth, "LZ4 (Depth)",
        lz4.frame.compress,
        lz4.frame.decompress
    )
    results.append(result)

    # Test LZMA if available
    if HAS_LZMA:
        print("Testing LZMA...")
        for preset in [1, 6, 9]:  # Fast, medium, best
            result = test_compression(
                rgb, f"LZMA-{preset} (RGB)",
                lambda data, p=preset: lzma.compress(data, preset=p),
                lzma.decompress,
                level=None
            )
            results.append(result)

        result = test_compression(
            depth, "LZMA-6 (Depth)",
            lambda data: lzma.compress(data, preset=6),
            lzma.decompress
        )
        results.append(result)

    # Test Zstandard if available
    if HAS_ZSTD:
        print("Testing Zstandard...")
        cctx = zstd.ZstdCompressor()
        dctx = zstd.ZstdDecompressor()

        for level in [3, 10, 19]:  # Fast, medium, best
            cctx = zstd.ZstdCompressor(level=level)
            result = test_compression(
                rgb, f"Zstd-{level} (RGB)",
                cctx.compress,
                dctx.decompress
            )
            results.append(result)

        cctx = zstd.ZstdCompressor(level=10)
        result = test_compression(
            depth, "Zstd-10 (Depth)",
            cctx.compress,
            dctx.decompress
        )
        results.append(result)

    # Test Blosc2 if available (optimized for numerical arrays)
    if HAS_BLOSC:
        print("Testing Blosc2...")
        result = test_compression(
            rgb, "Blosc2-LZ4 (RGB)",
            lambda data: blosc2.compress(data, codec=blosc2.Codec.LZ4),
            blosc2.decompress
        )
        results.append(result)

        result = test_compression(
            depth, "Blosc2-LZ4 (Depth)",
            lambda data: blosc2.compress(data, codec=blosc2.Codec.LZ4),
            blosc2.decompress
        )
        results.append(result)

    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"{'Algorithm':<20} {'Original':>10} {'Compressed':>12} {'Ratio':>8} "
          f"{'Compress':>12} {'Decompress':>12}")
    print(f"{'':20} {'(MB)':>10} {'(MB)':>12} {'':>8} {'(ms)':>12} {'(ms)':>12}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<20} {r['original_mb']:>10.2f} {r['compressed_mb']:>12.2f} "
              f"{r['ratio']:>8.2f}x {r['compress_time_ms']:>12.1f} {r['decompress_time_ms']:>12.1f}")

    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Calculate bandwidth savings
    print("For a typical 50-frame episode over WiFi:")
    print()

    lz4_size = sum(r['compressed_mb'] for r in results if 'LZ4' in r['name'] and 'Blosc' not in r['name'])

    for r in results:
        if r['name'] in ['LZMA-6 (RGB)', 'Zstd-10 (RGB)', 'Blosc2-LZ4 (RGB)']:
            # Find corresponding depth
            depth_name = r['name'].replace('RGB', 'Depth')
            depth_result = next((x for x in results if x['name'] == depth_name), None)
            if depth_result:
                total_size = r['compressed_mb'] + depth_result['compressed_mb']
                savings_mb = lz4_size - total_size
                savings_pct = (savings_mb / lz4_size) * 100
                total_compress_time = r['compress_time_ms'] + depth_result['compress_time_ms']

                print(f"{r['name'].split(' ')[0]}:")
                print(f"  Total size: {total_size:.2f} MB (saves {savings_mb:.2f} MB / {savings_pct:.1f}%)")
                print(f"  Compress time: {total_compress_time:.1f} ms (rover CPU time)")
                print(f"  Transfer time @ 50 Mbps: {total_size * 8 / 50 * 1000:.0f} ms")
                print()


if __name__ == '__main__':
    main()
