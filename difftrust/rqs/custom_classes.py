class ExperimentData:
    def __init__(self, logs, raw_dis, raw_err, ref_dis, ref_err, names, total):
        self.logs = logs
        self.raw_dis = raw_dis
        self.raw_err = raw_err
        self.ref_dis = ref_dis
        self.ref_err = ref_err
        self.names = names
        self.total = total

    def __repr__(self):
        return (f"ExperimentData(total={self.total}, "
                f"logs={len(self.logs)}, "
                f"raw_dis={len(self.raw_dis)}, ref_dis={len(self.ref_dis)})")


class ExperimentConfig:
    def __init__(self, llm, nb_candidate, nb_sample, temperature, timeout):
        self.llm = llm
        self.nb_candidate = nb_candidate
        self.nb_sample = nb_sample
        self.temperature = temperature
        self.timeout = timeout

    def __repr__(self):
        return (f"LLMConfig(llm={self.llm!r}, nb_candidate={self.nb_candidate}, "
                f"nb_sample={self.nb_sample}, temperature={self.temperature}, timeout={self.timeout})")

    @classmethod
    def from_dict(cls, data):
        return cls(
            llm=data["llm"],
            nb_candidate=data["nb_candidate"],
            nb_sample=data["nb_sample"],
            temperature=data["temperature"],
            timeout=data["timeout"]
        )

    def to_dict(self):
        return {
            "llm": self.llm,
            "nb_candidate": self.nb_candidate,
            "nb_sample": self.nb_sample,
            "temperature": self.temperature,
            "timeout": self.timeout
        }
