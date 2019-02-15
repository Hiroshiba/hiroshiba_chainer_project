from pathlib import Path

import chainer
from chainer import cuda

from project_name.config import Config
from project_name.model import create_predictor


class Generator(object):
    def __init__(
            self,
            config: Config,
            model_path: Path,
            gpu: int = None,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.gpu = gpu

        self.predictor = predictor = create_predictor(config.network)
        chainer.serializers.load_npz(str(model_path), predictor)

        if self.gpu is not None:
            predictor.to_gpu(self.gpu)
            cuda.get_device_from_id(self.gpu).use()

        chainer.global_config.train = False
        chainer.global_config.enable_backprop = False

    def generate(
            self,
    ):
        pass
