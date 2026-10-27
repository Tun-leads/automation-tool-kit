from __future__ import annotations

import asyncio
import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("automation")


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    name: str
    success: bool
    value: Any = None
    error: Exception | None = None
    elapsed: float = 0.0


@dataclass
class AutomationTask:
    name: str
    action: Callable[[], Any]
    retries: int = 0
    timeout: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.retries < 0:
            raise ValueError("retries must be zero or greater")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")

    def _run_action(self) -> Any:
        """Run the action, applying a timeout if one was configured."""

        if self.timeout is None:
            return self.action()

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.action)

        try:
            return future.result(timeout=self.timeout)
        finally:
            # A timeout cannot stop an action that is already running.
            # Do not wait for it here; it may finish in the background.
            executor.shutdown(wait=False, cancel_futures=True)

    def run(self) -> TaskResult:
        """Run a synchronous task with retries and an optional timeout."""

        start = time.perf_counter()

        for attempt in range(self.retries + 1):
            try:
                logger.info(
                    "Starting task '%s' - attempt %d",
                    self.name,
                    attempt + 1,
                )

                value = self._run_action()
                elapsed = time.perf_counter() - start

                logger.info("Completed task '%s'", self.name)

                return TaskResult(
                    name=self.name,
                    success=True,
                    value=value,
                    elapsed=elapsed,
                )

            except Exception as error:
                logger.warning(
                    "Task '%s' failed: %s",
                    self.name,
                    error,
                )

                if attempt == self.retries:
                    elapsed = time.perf_counter() - start

                    return TaskResult(
                        name=self.name,
                        success=False,
                        error=error,
                        elapsed=elapsed,
                    )

        raise RuntimeError("Unreachable code")


# ---------------------------------------------------------------------------
# Automation runner
# ---------------------------------------------------------------------------

class AutomationRunner:
    def __init__(self, max_workers: int = 4) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than zero")

        self.max_workers = max_workers

    def run_sequential(
        self,
        tasks: Iterable[AutomationTask],
    ) -> list[TaskResult]:
        """Run tasks one after another."""

        return [task.run() for task in tasks]

    def run_parallel(
        self,
        tasks: Iterable[AutomationTask],
    ) -> list[TaskResult]:
        """Run independent tasks concurrently using threads."""

        task_list = list(tasks)

        with ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            return list(executor.map(lambda task: task.run(), task_list))

    async def run_async(
        self,
        tasks: Iterable[AutomationTask],
    ) -> list[TaskResult]:
        """Run synchronous tasks concurrently, up to max_workers at a time."""

        task_list = list(tasks)
        semaphore = asyncio.Semaphore(self.max_workers)

        async def run_one(task: AutomationTask) -> TaskResult:
            async with semaphore:
                return await asyncio.to_thread(task.run)

        return await asyncio.gather(
            *(run_one(task) for task in task_list)
        )


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def automation_task(
    name: str,
    retries: int = 0,
    timeout: float | None = None,
) -> Callable[[Callable[[], Any]], AutomationTask]:
    """Convert a function into an AutomationTask."""

    def decorator(function: Callable[[], Any]) -> AutomationTask:
        return AutomationTask(
            name=name,
            action=function,
            retries=retries,
            timeout=timeout,
        )

    return decorator


def report_results(
    function: Callable[..., list[TaskResult]],
) -> Callable[..., list[TaskResult]]:
    """Decorator that prints a task summary."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> list[TaskResult]:
        results = function(*args, **kwargs)

        successful = sum(result.success for result in results)
        failed = len(results) - successful

        print(f"\nCompleted: {successful}")
        print(f"Failed: {failed}")

        for result in results:
            status = "OK" if result.success else "FAILED"
            print(f"[{status}] {result.name} ({result.elapsed:.2f}s)")

        return results

    return wrapper


# ---------------------------------------------------------------------------
# Example automation tasks
# ---------------------------------------------------------------------------

@automation_task("Create backup", retries=2)
def create_backup() -> str:
    time.sleep(1)
    return "backup.zip created"


@automation_task("Clean temporary files")
def clean_temp_files() -> str:
    time.sleep(0.5)
    return "temporary files removed"


@automation_task("Generate report")
def generate_report() -> str:
    time.sleep(0.8)
    return "report.html generated"


@automation_task("Send notification", retries=1)
def send_notification() -> str:
    time.sleep(0.4)
    return "notification sent"


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

@report_results
def run_automation() -> list[TaskResult]:
    runner = AutomationRunner(max_workers=4)

    tasks = [
        create_backup,
        clean_temp_files,
        generate_report,
        send_notification,
    ]

    return runner.run_parallel(tasks)


async def main() -> None:
    run_automation()

    print("\nRunning the same tasks asynchronously:")

    runner = AutomationRunner(max_workers=4)

    tasks = [
        create_backup,
        clean_temp_files,
        generate_report,
        send_notification,
    ]

    results = await runner.run_async(tasks)

    for result in results:
        if result.success:
            print(f"{result.name}: {result.value}")
        else:
            print(f"{result.name}: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())