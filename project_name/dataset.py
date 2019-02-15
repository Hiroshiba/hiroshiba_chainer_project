from functools import partial
from typing import Any, List

import chainer
import numpy

from project_name.config import DatasetConfig


class SampleDataset(chainer.dataset.DatasetMixin):
    def __init__(
            self,
            data_list: List[Any],
    ):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def get_example(self, i: int):
        return dict(
            input=self.data_list[i]['input'],
            target=self.data_list[i]['target'],
        )


def create(config: DatasetConfig):
    data_list = []
    numpy.random.RandomState(config.seed).shuffle(data_list)

    num_test = config.num_test
    trains = data_list[num_test:]
    tests = data_list[:num_test]
    evals = trains[:num_test]

    _Dataset = partial(
        SampleDataset,
        data_list=data_list,
    )
    return {
        'train': _Dataset(trains),
        'test': _Dataset(tests),
        'train_eval': _Dataset(evals),
    }
