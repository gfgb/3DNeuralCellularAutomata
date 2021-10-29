import json

class Params:
    def __init__(self, args):
        self.args = args
        self.args["use_pattern_pool"] = {"growing": False, "persistent": True, "regenerating": True}[self.experiment_type]
        self.args["damage_n"] = {"growing": 0, "persistent": 0, "regenerating": 3}[self.experiment_type]
        self.args["use_pretrained"] = False

    def __getattr__(self, item):
        return self.args[item]

    def add(self, param_name, value):
        self.args[param_name] = value

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.args, f, indent=4)

    @staticmethod
    def load(filename):
        with open(filename, "r") as f:
            params = Params(json.load(f))
        return params