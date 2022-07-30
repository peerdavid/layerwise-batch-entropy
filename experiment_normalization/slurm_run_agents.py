import subprocess
from math import ceil
from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Union
import warnings
from argparse import ArgumentParser, Namespace

import wandb
import yaml


slurm_template = Template(
    """#!/bin/bash -l
$sbatch_config
$cmd_str"""
)


def get_num_agents(full_sweep_id: str) -> int:
    api = wandb.Api()
    sweep = api.sweep(full_sweep_id)
    method = sweep.config["method"]
    if method != "grid":
        warnings.warn(f"Can only determine number of agents automatically for search method 'grid', not for '{method}'. Setting num_agents to 1")
        return 1
    
    num_agents = 1
    for param in sweep.config["parameters"].values():
        for key, val in param.items():
            if key == "values":
                num_agents *= len(val)
    return num_agents


def main(args: Namespace) -> None:
    config = load_config(args.sbatch_config)

    ntasks = int(config.get("ntasks", 1))
    ngpus = int(config.get("gres", ":1").split(":")[1])
    ntasks_per_gpu = ceil(ntasks / ngpus)

    sbatch_config = generate_sbatch_config(config)

    full_sweep_id = f"{args.entity}/{args.project}/{args.sweep_id}"
    cmd_str = f'srun -n {ntasks} --gpu-bind=single:{ntasks_per_gpu} wandb agent "{full_sweep_id}"'
    if args.count >= 0:
        cmd_str += f" --count {args.count}"

    num_agents = args.num_agents or get_num_agents(full_sweep_id)

    num_nodes = ceil(num_agents / ntasks)
    for i in range(num_nodes):
        content = slurm_template.substitute(
            sbatch_config=sbatch_config, cmd_str=cmd_str
        )
        with NamedTemporaryFile(mode="w", suffix=".sh") as bash_file:
            bash_file.write(content)
            bash_file.flush()

            print(f"Starting node: {i+1}/{num_nodes}")
            cmd = ["sbatch", bash_file.name]

            if args.test:
                print(cmd)
                print("")
                print(content)
               
                exit()
#                cmd.append("--test-only")

            subprocess.run(cmd)

#            if args.test:
#                break


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No config file found at {path.absolute()}")
    with path.open("r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    return config


def generate_sbatch_config(config: Dict[str, Any]) -> str:
    sbatch_config = ""
    for key, val in config.items():
        sbatch_config += f"#SBATCH --{key}={val}\n"

    return sbatch_config


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--entity", type=str, default="uibk_iis")
    parser.add_argument("--project", type=str, default="Layerwise Batch Entropy")
    parser.add_argument("sweep_id", type=str)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--num_agents", type=int, default=None, help="Number of wandb agents to start. When no value is given the number of agents is automatically determined, this only works for grid search!")
    parser.add_argument(
        "--sbatch_config", type=str, default="config/example_sbatch_config.yaml"
    )
    parser.add_argument("--test", action="store_true")

    main(parser.parse_args())
