import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union

from project_name.utility import JSONEncoder, namedtuple_to_dict


class DatasetConfig(NamedTuple):
    seed: int
    num_test: int


class NetworkConfig(NamedTuple):
    pass


class ModelConfig(NamedTuple):
    pass


class TrainConfig(NamedTuple):
    batchsize: int
    gpu: List[int]
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: Optional[int]
    optimizer: Dict[str, Any]
    optimizer_gradient_clipping: float
    linear_shift: Dict[str, Any]


class ProjectConfig(NamedTuple):
    name: str
    tags: List[str]


class Config(NamedTuple):
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    def save_as_json(self, path):
        d = namedtuple_to_dict(self)
        json.dump(d, open(path, 'w'), indent=2, sort_keys=True, cls=JSONEncoder)


def create_from_json(s: Union[str, Path]):
    return create(json.load(open(s)))


def create(d: Dict[str, Any]):
    backward_compatible(d)
    return Config(
        dataset=DatasetConfig(
            seed=d['dataset']['seed'],
            num_test=d['dataset']['num_test'],
        ),
        network=NetworkConfig(
        ),
        model=ModelConfig(
        ),
        train=TrainConfig(
            batchsize=d['train']['batchsize'],
            gpu=d['train']['gpu'],
            log_iteration=d['train']['log_iteration'],
            snapshot_iteration=d['train']['snapshot_iteration'],
            stop_iteration=d['train']['stop_iteration'],
            optimizer=d['train']['optimizer'],
            optimizer_gradient_clipping=d['train']['optimizer_gradient_clipping'],
            linear_shift=d['train']['linear_shift'],
        ),
        project=ProjectConfig(
            name=d['project']['name'],
            tags=d['project']['tags'],
        )
    )


def backward_compatible(d: Dict):
    pass
