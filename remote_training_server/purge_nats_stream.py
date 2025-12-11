#!/usr/bin/env python3
"""Purge old messages from NATS JetStream ROVER_EXPERIENCE stream."""

import asyncio
import nats

async def purge_stream():
    nc = await nats.connect("nats://nats.gokickrocks.org:4222")
    js = nc.jetstream()

    try:
        # Purge all messages from the stream
        await js.purge_stream("ROVER_EXPERIENCE")
        print("✅ Purged all messages from ROVER_EXPERIENCE stream")
    except Exception as e:
        print(f"❌ Error purging stream: {e}")

    await nc.close()

if __name__ == "__main__":
    asyncio.run(purge_stream())
