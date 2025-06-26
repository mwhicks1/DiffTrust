import random
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from difftrust.generic import generic_equal


def _safe_call(func, *args):
    try:
        return func(*args)
    except Exception as error:
        return error


def pointwise_incoherence(funcs: list[Callable], input_generator: Callable, samples: int):
    with ThreadPoolExecutor() as executor:

        futures = [executor.submit(input_generator) for _ in range(samples)]
        inputs = [future.result() for future in futures]

        futures_list = [
            [executor.submit(_safe_call, func, *args) for func in random.sample(funcs, 2)]
            for args in inputs
        ]

        dis = 0
        for future in futures_list:
            result1 = future[0].result()
            result2 = future[1].result()
            if not generic_equal(result1, result2):
                dis += 1
        return float(dis) / samples


def pointwise_error(funcs: list[Callable], ground_truth: Callable, input_generator: Callable, samples: int):
    with ThreadPoolExecutor() as executor:

        futures = [executor.submit(input_generator) for _ in range(samples)]
        inputs = [future.result() for future in futures]

        futures_list = [
            [
                executor.submit(_safe_call, random.choice(funcs), *args),
                executor.submit(_safe_call, ground_truth, *args)
            ] for args in inputs
        ]

        err = 0
        for future in futures_list:
            result1 = future[0].result()
            result2 = future[1].result()
            if not generic_equal(result1, result2):
                err += 1
        return float(err) / samples


def functional_incoherence(funcs: list[Callable], input_generator: Callable, samples: int):
    with ThreadPoolExecutor() as executor:

        futures = [executor.submit(input_generator) for _ in range(samples)]
        inputs = [future.result() for future in futures]

        futures_list = [
            [executor.submit(_safe_call, func, *args) for func in funcs]
            for args in inputs
        ]

        eq = [[True for _ in range(len(funcs))] for _ in range(len(funcs))]
        for future in futures_list:
            for i in range(len(funcs)):
                for j in range(i+1, len(funcs)):
                    result1 = future[i].result()
                    result2 = future[j].result()
                    if not generic_equal(result1, result2):
                        eq[i][j] = False
                        eq[j][i] = False

    visited = [False] * len(funcs)
    groups = []

    for i in range(len(funcs)):
        if not visited[i]:
            group = [i]
            visited[i] = True
            for j in range(i + 1, len(funcs)):
                if eq[i][j] and not visited[j]:
                    if all(eq[k][j] for k in group):
                        group.append(j)
                        visited[j] = True
            groups.append(group)

    func_dis = 0
    for group in groups:
        p = float(len(group)) / len(funcs)
        func_dis += p*(1-p)
    return func_dis


def functional_error(funcs: list[Callable], ground_truth: Callable, input_generator: Callable, samples: int):
    with ThreadPoolExecutor() as executor:

        futures = [executor.submit(input_generator) for _ in range(samples)]
        inputs = [future.result() for future in futures]

        futures_list = [
            [executor.submit(_safe_call, ground_truth, *args)] +
            [executor.submit(_safe_call, func, *args) for func in funcs]
            for args in inputs
        ]

        eq = [True for _ in range(len(funcs))]
        for future in futures_list:
            base_result = future[0].result()
            for i in range(len(funcs)):
                result = future[i+1].result()
                if not generic_equal(base_result, result):
                    eq[i] = False

    return float(sum(1 if not equal else 0 for equal in eq))/len(funcs)
