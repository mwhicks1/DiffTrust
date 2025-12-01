import time

from tqdm import tqdm

import difftrust
from HumanEval import instance


def compiling(candidate_list):
    candidate_list = [
        candidate.force_compile()
        for candidate in candidate_list
    ]

    return candidate_list


class ExperimentRunner:
    def __init__(self, exp_ctxt: difftrust.experiment.ExperimentCtxt, dataset: instance.Dataset):
        self.ctxt = exp_ctxt
        self.dataset = dataset
        self.llm = difftrust.llm.get_llm_by_name(self.ctxt.get_config("llm"))
        self.llm.temperature = self.ctxt.get_config("temperature")
        self.coder = difftrust.coder.Coder(self.llm)
        self.spec_refiner = difftrust.refiner.SpecificationRefiner(self.llm)
        self.nb_candidate = self.ctxt.get_config("nb_candidate")
        self.nb_sample = self.ctxt.get_config("nb_sample")
        self.timeout = self.ctxt.get_config("timeout")

    def seen(self, inst: instance.Instance, typ: str, suffix):
        exp_name = f"experiment-{typ}_{suffix}" if suffix != "" else f"experiment-{typ}"
        logs = self.ctxt.get_logs(exp_name)
        seen = {log["task_id"] for log in logs}
        cache_file = f"candidates_{inst.task_id}".replace("/", "_")
        candidate_list = self.ctxt.get_cache(cache_file)
        size_requirement = False
        if candidate_list:
            size_requirement = self.nb_candidate <= len(candidate_list)

        return (inst.task_id in seen and size_requirement) or (inst.task_id in seen and suffix != "")

    def get_log(self, inst: instance.Instance):
        logs = self.ctxt.get_logs("experiment")
        return [log for log in logs if logs["task_id"] == inst.task_id][0]

    def get_inst(self, task_id: str):
        return [inst for inst in self.dataset.instances if inst.task_id == task_id][0]

    def refined_spec(self, inst: instance.Instance) -> difftrust.specification.ProbabilisticSpecification:
        cache_file = f"refined_spec_{inst.task_id.replace('/', '_')}"
        refined_spec = self.ctxt.get_cache(cache_file)
        if refined_spec is None:
            refined_spec = self.spec_refiner.refine(inst.spec)
            self.ctxt.cache_object(cache_file, refined_spec)
        return refined_spec

    def candidate_list(self, inst: instance.Instance, call_llm_api=True, slow_mode=False) -> list[
        difftrust.function.Function]:
        cache_file = f"candidates_{inst.task_id.replace('/', '_')}"
        candidate_list = self.ctxt.get_cache(cache_file)
        if call_llm_api:
            candidate_list = [] if candidate_list is None else candidate_list
            number_of_samples_needed = max(0, self.nb_candidate - len(candidate_list))
            print(f"Getting {number_of_samples_needed} candidates")
            if slow_mode:
                for i in range(number_of_samples_needed):
                    candidate_list.extend(self.coder.sample(inst.spec, 1, slow_mode=slow_mode))
                    print(f"Caching updated candidate list. Current length {len(candidate_list)}")
                    self.ctxt.cache_object(cache_file, candidate_list)
                    time.sleep(1)
            else:
                candidate_list.extend(self.coder.sample(inst.spec, number_of_samples_needed))
            self.ctxt.cache_object(cache_file, candidate_list)
        print(f"Returning {self.nb_candidate}/{len(candidate_list)} candidates.")
        return candidate_list[:self.nb_candidate]

    def run_one_instance(self, inst: instance.Instance, typ: str, call_llm_api, suffix="", slow_mode=False):
        print(f"Type is {typ}")
        experiment_log_name = f"experiment-{typ}_{suffix}" if suffix != "" else f"experiment-{typ}"
        if typ == "functional":
            compute_disagreement = difftrust.disagreement.functional_incoherence
            compute_error = difftrust.disagreement.functional_error
        else:
            compute_disagreement = difftrust.disagreement.pointwise_incoherence
            compute_error = difftrust.disagreement.pointwise_error

        print(f"-- {inst.name} (task_id={inst.task_id}) --")
        if self.seen(inst, typ, suffix):
            print(f"Done.")
            return

        t = time.time()
        log = {"name": inst.name, "task_id": inst.task_id}

        try:
            candidate_list = self.candidate_list(inst, call_llm_api, slow_mode=slow_mode)
        except Exception as exception:
            msg = (f"Exception occurred during when querying the LLM -- "
                   f"{type(exception).__name__} : {exception}")
            print(msg)
            log["Failed"] = msg
            self.ctxt.log(experiment_log_name, log)
            return

        print("Compiling...")
        try:
            candidate_list = difftrust.checking.timeout_call(
                compiling,
                (candidate_list,),
                {},
                self.timeout
            )
        except Exception as exception:
            msg = (f"Exception occurred during compilation of LLM-generated code -- "
                   f"{type(exception).__name__} : {exception}")
            print(msg)
            log["Failed"] = msg
            self.ctxt.log(experiment_log_name, log)
            return

        print("Computing statistics...")

        try:
            if len(candidate_list) == 1:
                log["Dis"] = 0.0
            else:
                log["Dis"] = difftrust.checking.timeout_call(
                    func=compute_disagreement,
                    args=(candidate_list, inst.filtered_generator, self.nb_sample),
                    kwargs={},
                    timeout=self.timeout
                )
            print(f"Disagreement : {log['Dis']}")
        except TimeoutError:
            msg = f"Timeout of {self.timeout} s. has been hit during disagreement computation"
            print(msg)
            log["Failed"] = msg
            self.ctxt.log(experiment_log_name, log)
            return
        except Exception as e:
            import traceback
            msg = f"An exception ({type(e).__name__} : {e}) occurred during disagreement computation"
            print(msg)
            print("Full traceback:")
            traceback.print_exc()
            log["Failed"] = msg
            self.ctxt.log(experiment_log_name, log)
            return

        try:
            log["Err"] = difftrust.checking.timeout_call(
                func=compute_error,
                args=(candidate_list, inst.ground_truth, inst.filtered_generator, self.nb_sample),
                kwargs={},
                timeout=self.timeout
            )
            print(f"Error : {log['Err']}")
        except TimeoutError:
            msg = f"Timeout of {self.timeout} s. has been hit during error computation"
            print(msg)
            log["Failed"] = msg
            self.ctxt.log(experiment_log_name, log)
            return
        except Exception as e:
            msg = f"An exception ({type(e).__name__} : {e}) occurred during error computation"
            print(msg)
            log["Failed"] = msg
            self.ctxt.log(experiment_log_name, log)
            return

        log["TotalTime"] = time.time() - t
        self.ctxt.log(experiment_log_name, log)

    def run(self, typ: str, call_llm_api, suffix, slow_mode):
        print("Querying LLMs and computing disagreement...")
        for inst in tqdm(self.dataset.instances):
            self.run_one_instance(inst, typ, call_llm_api, suffix, slow_mode)


if __name__ == "__main__":
#    llm_name = "gemini_2_0_flash"
    llm_name = "claude_3_5_sonnet_bedrock"  # or any other bedrock model
    slow_mode = True
    print(f"Model is {llm_name}")
    temp = 0.2
    ctxt = difftrust.experiment.ExperimentCtxt(f".experiments/AAAA")
    ctxt.set_config("llm", llm_name)
    ctxt.set_config("nb_candidate", 10)
    ctxt.set_config("nb_sample", 1000)
    ctxt.set_config("temperature", temp)
    ctxt.set_config("timeout", 60.0)
    print(f"Model is {llm_name} temp is {ctxt.get_config("temperature")}")
    dataset = instance.Dataset("dataset-v1")
    # dataset.make()
    dataset.load()
    # new_dataset = copy(dataset)
    # new_instances = regenerate(dataset.instances)
    # new_dataset.instances = new_instances

    runner = ExperimentRunner(ctxt, dataset)

#    runner.run_one_instance(dataset.instances[0], "pointwise", call_llm_api=True, suffix="", slow_mode=None)
    runner.run("pointwise", call_llm_api=True, suffix="", slow_mode=None)
