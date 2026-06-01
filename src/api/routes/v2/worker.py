from __future__ import annotations

import asyncio
import logging

from src.api.routes.v2.activities import ALL_ACTIVITIES
from src.api.routes.v2.temporal_client import get_temporal_client
from src.api.routes.v2.workflows import ALL_WORKFLOWS
from src.config import settings

logger = logging.getLogger("xmem.api.routes.v2.worker")


async def main() -> None:
    try:
        from temporalio.worker import Worker
    except Exception as exc:
        raise RuntimeError("temporalio is required to run the v2 worker") from exc

    client = await get_temporal_client()
    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=ALL_WORKFLOWS,
        activities=ALL_ACTIVITIES,
    )
    logger.info("Starting XMem v2 Temporal worker on %s", settings.temporal_task_queue)
    await worker.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
