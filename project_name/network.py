from typing import Union

import chainer
import numpy


class SampleNetwork(chainer.Chain):
    def __init__(self):
        super().__init__()

        with self.init_scope():
            pass

    def __call__(self, xs: Union[chainer.Variable, numpy.ndarray]):
        """
        :param xs: shape: (length, ?)
        """
        return xs
