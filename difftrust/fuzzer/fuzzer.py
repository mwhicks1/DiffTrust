import importlib.util
import math
import os
import random
import shutil
import sys
import tempfile
from typing import Callable

from difftrust.generic import generic_repr
from difftrust.generic import generic_mutator
from difftrust.fuzzer.coverage import LineCoverageTracer


def entropy(scores: dict):
    scores = list(scores.values())
    total = sum(score for score in scores)
    if total < 10:
        return float((10 - total) + 1)
    else:
        probs = [float(score) / total for score in scores]
        return sum(-prob * math.log(prob) for prob in probs) + 1.0


def sample(lst: list[tuple]):
    total = sum(score for _, score in lst)
    rnd = random.uniform(0.0, total)
    count = 0.0
    for elt, score in lst:
        count += score
        if count >= rnd:
            return elt
    assert 0


class Scheduler:
    def __init__(self):
        self.total_seen_path = 0
        self.path2count = {}
        self.inputs2paths = {}

    def _increment_count(self, path):
        self.total_seen_path += 1
        self.path2count[path] = self.path2count.get(path, 0) + 1

    def produced_path(self, inputs, path):
        self._increment_count(path)
        paths = self.inputs2paths.get(id(input), set())
        paths.add(path)
        self.inputs2paths[id(inputs)] = paths

    def _path_power(self, path):
        return 1.0 / (self.path2count.get(path, 0) + 1)

    def power(self, inputs):
        paths = self.inputs2paths.get(id(inputs), set())
        return math.exp(10 * sum(self._path_power(path) for path in paths))


class Fuzzer:
    def __init__(self, func: Callable, corpus: list[tuple]):
        self.tracer = LineCoverageTracer()
        self.func = func
        self.scheduler = Scheduler()
        self.corpus = corpus
        self.covered = {}
        for idx in range(len(corpus)):
            result, path = self.tracer.run(self.func, *corpus[idx])
            self.covered[path] = idx

    def draw_idx(self):
        tmp = [
            (idx, self.scheduler.power(self.corpus[idx])) for idx in range(len(self.corpus))
        ]
        return sample(tmp)

    def mutate(self, idx):
        return generic_mutator(self.corpus[idx], random.randint(1, 10))

    def one_fuzz(self, idx: int):
        # print("Testing : ", self.corpus[idx])
        inputs = self.mutate(idx)

        output, path = self.tracer.run(self.func, *inputs)
        self.scheduler.produced_path(self.corpus[idx], path)

        if path not in self.covered:
            self.covered[path] = len(self.corpus)
            self.corpus.append(inputs)
            print(inputs)
        else:
            idx = self.covered[path]
            if len(generic_repr(inputs)) < len(generic_repr(self.corpus[idx])):
                print(self.corpus[idx], " -> ", inputs)
                self.covered[path] = len(self.corpus)
                self.corpus.append(inputs)

    def fuzz(self, iters):
        for _ in range(iters):
            self.one_fuzz(self.draw_idx())
