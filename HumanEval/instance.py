import inspect
import json
import pathlib
import random
from typing import Callable

import cloudpickle

import difftrust


class Instance:
    generator_nb_try: int = 100

    def __init__(self, info: dict):
        self.task_id = info["task_id"]
        self.prompt = info["prompt"]
        self.entry_point = info["entry_point"]
        self.canonical_solution = info["canonical_solution"]
        self.test = info["test"]
        self.code = self.prompt + self.canonical_solution

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

    def make_name(self):
        return self.entry_point

    def make_description(self):
        txt = self.name.join(self.prompt.split(self.name)[1:])
        return txt[txt.find(':\n') + 1:]

    def make_ground_truth(self) -> Callable:
        namespace = {}
        exec(self.code, namespace)
        return namespace[self.name]

    # Do not remove this if you want to use our generated instances
    def print_hi(self):
        print("hi")

    def make_spec(self):
        name = self.name
        description = self.description
        signature = str(inspect.signature(self.ground_truth))
        return difftrust.specification.Specification(
            name=name,
            signature=signature,
            description=description
        )

    def make_inputs_corpus(self):
        input_corpus = []

        def smart_candidate(*args):
            input_corpus.append(args)
            return self.ground_truth(*args)

        code = (
            f"{self.test}\n"
            f"\n"
            f"check(smart_candidate)"
        )

        try:
            exec(code)
        except Exception:
            pass

        return input_corpus

    def check_validity(self):
        """ Check the validity of the instance """

        # Checking the validity of the ground truth with respect to the asserts

        check_code = (
            f"{self.test}\n"
            f"\n"
            f"check(self.ground_truth)"
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
        return self.check_validity() and self.check_speed()


class Dataset:
    def __init__(self, name: str):
        self.name = name
        self.instances = []
        self.data_path = pathlib.Path(__file__).parent / ".data"
        if not (self.data_path.exists() and (self.data_path / "human-eval-v2-20210705.jsonl").exists()):
            raise Exception("The folder .data containing sanitized-mbpp.json does not exists !")

    def load(self):
        with open(f"{self.data_path.as_posix()}/{self.name}.pkl", "rb") as f:
            self.instances = cloudpickle.load(f)

    def save(self):

        with open(f"{self.data_path.as_posix()}/{self.name}.pkl", "wb") as f:
            cloudpickle.dump(self.instances, f)

    def make(self):

        with open(f"{self.data_path.as_posix()}/human-eval-v2-20210705.jsonl", "r", encoding="utf-8") as f:
            instance_list = [
                json.loads(line) for line in f.readlines()
            ]
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
