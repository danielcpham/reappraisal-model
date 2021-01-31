from pathlib import Path

class LDHConfig:
    def __init__(self, training_dir=None, eval_dir=None) -> None:

        DEFAULT_CONFIG = {
            "train_dir": Path.cwd() / 'src' / 'training',
            "eval_dir": Path.cwd() / 'src' / 'eval'
        }
        self._train_dir = training_dir if training_dir else DEFAULT_CONFIG['train_dir']
        self._eval_dir = eval_dir if eval_dir else DEFAULT_CONFIG['eval_dir']

    @property
    def train_dir(self) -> Path:
        return self._train_dir

    @property
    def eval_dir(self) -> Path:
        return self._eval_dir

    @staticmethod
    def make():
        return LDHConfig()