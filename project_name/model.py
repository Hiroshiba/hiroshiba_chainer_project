import chainer
import chainer.functions as F
from chainer import Chain

from project_name.config import ModelConfig, NetworkConfig
from project_name.network import SampleNetwork


def create_predictor(config: NetworkConfig):
    predictor = SampleNetwork(
    )
    return predictor


class Model(Chain):
    def __init__(self, model_config: ModelConfig, predictor: SampleNetwork) -> None:
        super().__init__()
        self.model_config = model_config
        with self.init_scope():
            self.predictor = predictor

    def __call__(
            self,
            input: chainer.Variable,
            target: chainer.Variable,
    ):
        output = self.predictor(input)
        loss = F.softmax_cross_entropy(output, target)

        chainer.report(dict(
            loss=loss,
        ), self)
        return loss
