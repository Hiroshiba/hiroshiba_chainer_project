import re
from pathlib import Path

import fire
import numpy

from project_name.config import create_from_json as create_config
from project_name.utility import save_arguments


def _extract_number(f):
    s = re.findall(r'\d+', str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
        model_dir: Path,
        iteration: int = None,
        prefix: str = 'main_',
):
    if iteration is None:
        paths = model_dir.glob(prefix + '*.npz')
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + '{}.npz'.format(iteration))
    return model_path


def generate(
        model_dir_path: str,
        model_iteration: int,
        model_config_path: str,
        output_dir_path: str = './output/',
        gpu: int = None,
):
    model_dir = Path(model_dir_path)
    output_dir = Path(output_dir_path)

    output_dir.mkdir(exist_ok=True)

    output = output_dir / model_dir.name
    output.mkdir(exist_ok=True)

    save_arguments(generate, locals(), output / 'arguments.json')

    config = create_config(model_config_path)
    model_path = _get_predictor_model_path(
        model_dir=model_dir,
        iteration=model_iteration,
    )

    data_list = []
    numpy.random.RandomState(config.dataset.seed).shuffle(data_list)
    texts = data_list[:config.dataset.num_test]

    pass


if __name__ == '__main__':
    fire.Fire(generate)
