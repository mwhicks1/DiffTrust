import inspect
import json
import pathlib
import random
import re
from typing import Callable

import cloudpickle

import difftrust

# Number of try of the instance generator
_NB_TRY = 100


def _extract_inputs(inputs_str: str):
    count = 0
    for idx in range(len(inputs_str)):
        if inputs_str[idx] == '(':
            count += 1
        elif inputs_str[idx] == ')':
            count -= 1

        if count == 0:
            return idx + 1
    assert 0


class Instance:
    generator_nb_try: int = 100

    def __init__(self, info: dict):
        self.task_id = info["task_id"]
        self.prompt = info["prompt"]
        self.code = info["code"]
        self.test_imports = info["test_imports"]
        self.test_list = info["test_list"]

        # Attributes
        self.name = self.make_name()
        self.description = self.make_description()
        self.ground_truth = self.make_ground_truth()
        self.spec = self.make_spec()
        self.inputs_corpus = self.make_inputs_corpus()

    def blind_generator(self):
        inputs = random.choice(self.inputs_corpus)
        return tuple(difftrust.generic.generic_mutator(x, random.choice([1, 10, 30])) for x in inputs)

    def filtered_generator(self):
        for i in range(self.generator_nb_try):
            inputs = random.choice(self.inputs_corpus)
            inputs = tuple(difftrust.generic.generic_mutator(x, random.choice([1, 10, 30])) for x in inputs)
            try:
                self.ground_truth(*inputs)
                return inputs
            except Exception:
                pass

        return random.choice(self.inputs_corpus)

    def print_hi(self):
        print("hi!")

    def make_name(self):
        tests = '\n'.join(self.test_list)
        func_names = re.findall(r"def\s+(\w+)\s*\(.*\)\s*:", self.code)
        for func_name in func_names:
            if func_name in tests:
                return func_name
        print(func_names)
        print(self.code)

    def make_description(self):
        description = self.prompt.replace("Write a python function to ", "")
        description = description.replace("Write a function to ", "")
        return description

    def make_ground_truth(self) -> Callable:
        name = self.make_name()
        namespace = {}
        exec(self.code, namespace)
        return namespace[name]

    def make_spec(self):
        name = self.make_name()
        description = self.make_description()
        signature = str(inspect.signature(self.make_ground_truth()))
        return difftrust.specification.Specification(
            name=name,
            signature=signature,
            description=description
        )

    def make_inputs_corpus(self):
        inputs_list = []
        for test in self.test_list:
            inputs_str = re.search(r"" + self.make_name() + r"\s*(\(.*\))", test).group(1)
            inputs_str = inputs_str[:_extract_inputs(inputs_str)]
            code = (
                f"{'\n'.join(self.test_imports)}\n"
                f"inputs__ = {inputs_str}"
            )
            namespace = {}
            exec(code, namespace)
            inputs = namespace["inputs__"]
            if not isinstance(inputs, tuple) and len(inspect.signature(self.ground_truth).parameters) == 1:
                inputs = (inputs,)
            elif isinstance(inputs, tuple) and len(inspect.signature(self.ground_truth).parameters) != len(inputs):
                inputs = (inputs,)
            inputs_list.append(inputs)
        return inputs_list

    def check_validity(self):
        """ Check the validity of the instance """

        # Checking the validity of the ground truth with respect to the asserts

        check_code = (
            f"{self.make_name()}=self.ground_truth\n"
            f"{'\n'.join(self.test_imports)}\n"
            f"{'\n'.join(self.test_list)}"
        )

        try:
            exec(check_code)
        except Exception:
            return False

        # Checking the speed of the generator

        return True

    def check_speed(self):
        def func(*args):
            try:
                self.ground_truth(*args)
            except Exception as e:
                return e

        return difftrust.checking.check_speed(func, self.blind_generator, 10000, 10.0)

    def check(self):
        return self.check_speed() and self.check_validity()


class Dataset:
    def __init__(self, name: str):
        self.name = name
        self.instances = []
        self.data_path = pathlib.Path(__file__).parent / ".data"
        if not (self.data_path.exists() and (self.data_path / "sanitized-mbpp.json").exists()):
            raise Exception("The folder .data containing sanitized-mbpp.json does not exists !")

    def load(self):
        with open(f"{self.data_path.as_posix()}/{self.name}.pkl", "rb") as f:
            self.instances = cloudpickle.load(f)

    def save(self):

        with open(f"{self.data_path.as_posix()}/{self.name}.pkl", "wb") as f:
            cloudpickle.dump(self.instances, f)

    def make(self):

        with open(f"{self.data_path.as_posix()}/sanitized-mbpp.json", "r", encoding="utf-8") as f:
            instance_list = json.load(f)
        instance_list = [Instance(instance) for instance in instance_list]

        self.instances = []
        for inst in instance_list:
            accepted = inst.check()
            print(f"{inst.name} (task_id={inst.task_id}) : {accepted}")
            if accepted:
                self.instances.append(inst)
                self.save()


if __name__ == "__main__":
    dataset = Dataset("dataset-v1")
    dataset.load()
    dataset.save()
