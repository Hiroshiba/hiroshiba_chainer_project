"""
launcher for some task that have diff params
"""

import argparse
import copy
import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

base_command_default = \
    "tmux new -ds {project/name}_gpu{gpu} ;" + \
    "tmux send -t {project/name}_gpu{gpu} 'pipenv run python {python_file_path} {recipe_path} {output}\n'"

parser = argparse.ArgumentParser()
parser.add_argument('output_dir', type=Path)
parser.add_argument('--python_file_path', default='train.py')
parser.add_argument('--recipe_json_path', default='recipe/recipe.json')
parser.add_argument('--base_config_json_path', default='recipe/config.json')
parser.add_argument('--base_command', default=base_command_default)
args = parser.parse_args()

recipe = json.load(open(args.recipe_json_path, encoding='utf-8'))
recipe_each: Dict[str, Any] = recipe['each']
recipe_all: Dict[str, Any] = recipe['all']
base_config = json.load(open(args.base_config_json_path, encoding='utf-8'))


def put_config_value(config, recipe_key, value):
    key_tree = recipe_key.split('/')
    target = config
    for key in key_tree[:-1]:
        target = target[key]

    target[key_tree[-1]] = value


def make_key_chain(key_chain, value, dist):
    if not isinstance(value, dict):
        dist['/'.join(key_chain)] = value
    else:
        for key in value.keys():
            make_key_chain(key_chain + [key], value[key], dist)


def replace_name(config):
    _format = {}
    make_key_chain([], config, _format)

    now = datetime.datetime.now()
    _format['date'] = now.strftime('%Y%m%d%H%M%S')
    _format['hash'] = hashlib.md5(bytes(str(now), 'utf')).hexdigest()[:6]

    config['project']['name'] = config['project']['name'].format(**_format)


num_task = min(len(list(value)) for value in recipe_each.values())
command_list = []

for i in range(num_task):
    config = copy.deepcopy(base_config)

    for recipe_key in recipe_all.keys():
        put_config_value(config, recipe_key, recipe_all[recipe_key])

    for recipe_key in recipe_each.keys():
        put_config_value(config, recipe_key, recipe_each[recipe_key][i])

    made_recipe_path = "{}.{}.json".format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'), i)
    with open(made_recipe_path, 'w', encoding='utf') as f:
        json.dump(config, f, indent=2, sort_keys=True, ensure_ascii=False)

    replace_name(config)

    dist = {}
    make_key_chain([], config, dist)

    dist['output'] = args.output_dir / config['project']['name']
    dist['python_file_path'] = args.python_file_path
    dist['recipe_path'] = made_recipe_path
    dist['gpu'] = config['train']['gpu']

    command = args.base_command.format(**dist)
    command_list += [command]

    print(config['project']['name'])

for command in command_list:
    subprocess.check_output(command, shell=True)
