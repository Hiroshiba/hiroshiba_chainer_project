from copy import copy
from pathlib import Path
from typing import Any, Dict

import fire
from chainer import cuda, optimizer_hooks, optimizers, training
from chainer.iterators import MultiprocessIterator
from chainer.training import extensions
from chainer.training.updaters import StandardUpdater
from tb_chainer import SummaryWriter

from project_name.config import create_from_json
from project_name.dataset import create as create_dataset
from project_name.model import Model, create_predictor
from utility import TensorBoardReport


def train(
        config_json_path: str,
        output_path: str,
):
    config_json = Path(config_json_path)
    output = Path(output_path)

    config = create_from_json(config_json)
    output.mkdir(exist_ok=True, parents=True)
    config.save_as_json((output / 'config.json').absolute())

    # model
    predictor = create_predictor(config.network)
    model = Model(model_config=config.model, predictor=predictor)

    if config.train.gpu is not None:
        model.to_gpu(config.train.gpu)
        cuda.get_device_from_id(config.train.gpu).use()

    # dataset
    dataset = create_dataset(config.dataset)
    train_iter = MultiprocessIterator(dataset['train'], config.train.batchsize, repeat=True, shuffle=True)
    test_iter = MultiprocessIterator(dataset['test'], config.train.batchsize, repeat=False, shuffle=False)
    train_eval_iter = MultiprocessIterator(dataset['train_eval'], config.train.batchsize, repeat=False, shuffle=False)

    # optimizer
    def create_optimizer(model):
        cp: Dict[str, Any] = copy(config.train.optimizer)
        n = cp.pop('name').lower()

        if n == 'adam':
            optimizer = optimizers.Adam(**cp)
        elif n == 'sgd':
            optimizer = optimizers.SGD(**cp)
        else:
            raise ValueError(n)

        optimizer.setup(model)

        if config.train.optimizer_gradient_clipping is not None:
            optimizer.add_hook(optimizer_hooks.GradientClipping(config.train.optimizer_gradient_clipping))

        return optimizer

    optimizer = create_optimizer(model)

    # updater
    updater = StandardUpdater(
        iterator=train_iter,
        optimizer=optimizer,
        device=config.train.gpu,
    )

    # trainer
    trigger_log = (config.train.log_iteration, 'iteration')
    trigger_snapshot = (config.train.snapshot_iteration, 'iteration')
    trigger_stop = (config.train.stop_iteration, 'iteration') if config.train.stop_iteration is not None else None

    trainer = training.Trainer(updater, stop_trigger=trigger_stop, out=str(output))
    tb_writer = SummaryWriter(output)

    if config.train.linear_shift is not None:
        trainer.extend(extensions.LinearShift(**config.train.linear_shift))

    ext = extensions.Evaluator(test_iter, model, device=config.train.gpu)
    trainer.extend(ext, name='test', trigger=trigger_log)
    ext = extensions.Evaluator(train_eval_iter, model, device=config.train.gpu)
    trainer.extend(ext, name='train', trigger=trigger_log)

    ext = extensions.snapshot_object(predictor, filename='main_{.updater.iteration}.npz')
    trainer.extend(ext, trigger=trigger_snapshot)

    trainer.extend(extensions.FailOnNonNumber(), trigger=trigger_log)
    trainer.extend(extensions.observe_lr(), trigger=trigger_log)
    trainer.extend(extensions.LogReport(trigger=trigger_log))
    trainer.extend(extensions.PrintReport(['main/model', 'test/main/model']), trigger=trigger_log)
    trainer.extend(TensorBoardReport(writer=tb_writer), trigger=trigger_log)
    if trigger_stop is not None:
        trainer.extend(extensions.ProgressBar(trigger_stop))

    trainer.run()


if __name__ == '__main__':
    fire.Fire(train)
