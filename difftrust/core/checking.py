import multiprocessing
import cloudpickle
from typing import Callable
from queue import Empty


def _timeout_worker(input_queue, output_queue):
    while True:
        task = input_queue.get()
        if task is None:
            return
        task = cloudpickle.loads(task)
        result = task()
        output_queue.put(cloudpickle.dumps(result))


def timeout_call(func: Callable, args: tuple, kwargs: dict, timeout: float):
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    worker = multiprocessing.Process(target=_timeout_worker, args=(input_queue, output_queue))
    worker.start()
    task = cloudpickle.dumps(lambda: func(*args, **kwargs))
    input_queue.put(task)

    try:
        result = cloudpickle.loads(output_queue.get(timeout=timeout))
        return result
    except Empty:
        raise TimeoutError()
    finally:
        # Gracefully stop worker and clean up resources
        try:
            input_queue.put(None)
        except:
            pass  # If queue is broken
        worker.terminate()
        worker.join()
        input_queue.close()
        input_queue.join_thread()
        output_queue.close()
        output_queue.join_thread()


def check_timeout(func: Callable, generator: Callable, timeout: float, tests: int):
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    worker = multiprocessing.Process(target=_timeout_worker, args=(input_queue, output_queue))
    worker.start()

    try:
        for _ in range(tests):
            inputs = generator()
            task = cloudpickle.dumps(lambda: func(*inputs))
            input_queue.put(task)
            try:
                output_queue.get(timeout=timeout)
            except Empty:
                return False
        return True
    finally:
        # Tell worker to exit and clean up
        try:
            input_queue.put(None)
        except:
            pass
        worker.terminate()
        worker.join()
        input_queue.close()
        input_queue.join_thread()
        output_queue.close()
        output_queue.join_thread()


def _speed_worker(input_queue, output_queue):
    func = input_queue.get()
    func = cloudpickle.loads(func)

    while True:

        inputs = input_queue.get()
        if inputs is None:
            output_queue.put(True)
            return

        try:
            func(*inputs)
        except Exception:
            pass


def check_speed(func: Callable, generator: Callable, execs: int, duration: float):
    """ Check that func is faster that execs/duration """
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    worker = multiprocessing.Process(target=_speed_worker, args=(input_queue, output_queue))
    worker.start()
    input_queue.put(cloudpickle.dumps(func))
    for test in range(execs):
        inputs = generator()
        input_queue.put(inputs)
    input_queue.put(None)
    try:
        output_queue.get(timeout=duration)
    except Empty:
        worker.terminate()
        worker.join()
        return False
    worker.terminate()
    worker.join()
    return True


def check_prop(func: Callable, generator: Callable, prop: Callable, tests: int):
    """ Return a set of Exceptions encountered """
    for test in range(tests):
        inputs = generator()
        try:
            output = func(*inputs)
        except Exception as exception:
            output = exception
        if not prop(inputs, output):
            return False
    return True
