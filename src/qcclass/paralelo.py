"""Run independent tasks sequentially or in a process pool, with a progress bar."""

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


def run_tasks[T, R](
    function: Callable[[T], R], tasks: Sequence[T], processes: int, description: str
) -> list[R]:
    """Apply ``function`` to every task; results are returned in completion order.

    ``function`` must be picklable (a module-level function or a ``functools.partial`` of
    one) when ``processes > 1``.
    """
    results: list[R] = []
    with tqdm(total=len(tasks), desc=description, unit="task") as progress:
        if processes <= 1:
            for task in tasks:
                results.append(function(task))
                progress.update()
        else:
            with ProcessPoolExecutor(max_workers=processes) as pool:
                futures = [pool.submit(function, task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
                    progress.update()
    return results
