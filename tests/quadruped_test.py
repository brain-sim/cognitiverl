import asyncio

from isaacsim.examples.interactive.quadruped import QuadrupedExample


async def test():
    # Create the QuadrupedExample instance
    example = QuadrupedExample()

    # Initialize the world (equivalent to clicking "Load" in the UI)
    await example.load_world_async()


asyncio.ensure_future(test())
