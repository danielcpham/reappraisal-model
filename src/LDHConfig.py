
class LDHConfig:
    def __init__(self, training_dir=None, eval_dir=None) -> None:

        DEFAULT_CONFIG =  {
            "training_dir": "./src/training",
            "eval_dir": "./src/test"
        }
        self._training_dir = training_dir if training_dir else DEFAULT_CONFIG['training_dir']
        self._eval_dir = eval_dir if eval_dir else DEFAULT_CONFIG['eval_dir']

    @property
    def training_dir(self):
        return self._training_dir

    @property
    def eval_dir(self):
        return self._eval_dir

    @staticmethod
    def make():
        return LDHConfig()