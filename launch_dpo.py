"""
Launcher script for `run_dpo.py` that takes care of setting up distributed training through deepspeed.
To run locally:

    python launch_dpo.py --config dpo_config/example.yaml --working $WORKING_DIR

In addition, the script also supports submitting jobs through slurm by using the --gpus argument.
Multi-node training is also supported. For instance, the following command would launch a multi-node job
on 2 nodes (each with 8 GPUs):

    python launch_dpo.py --config dpo_config/example.yaml --working $WORKING_DIR --gpus 16
"""
import argparse
import os
import subprocess
import sys

import submitit
import yaml

GPUS_PER_NODE = 8


def dict2args(d):
    args = []
    for k, v in d.items():
        args.append(f"--{k}")
        if isinstance(v, list):
            for x in v:
                args.append(str(x))
        else:
            args.append(str(v))
    return args


def dpo_task(nodes, config):
    env = submitit.helpers.TorchDistributedEnvironment()
    ds_config = {
        "compute_environment": "LOCAL_MACHINE",
        "debug": False,
        "deepspeed_config": {
            "deepspeed_multinode_launcher": "standard",
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "offload_optimizer_device": "none",
            "offload_param_device": "none",
            "zero3_init_flag": False,
            "zero_stage": 2,
        },
        "distributed_type": "DEEPSPEED",
        "downcast_bf16": "no",
        "machine_rank": env.rank,
        "main_process_ip": env.master_addr,
        "main_process_port": env.master_port,
        "main_training_function": "main",
        "mixed_precision": "bf16",
        "num_machines": nodes,
        "num_processes": nodes * GPUS_PER_NODE,
        "rdzv_backend": "static",
        "same_network": True,
        "tpu_env": [],
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": False,
    }
    config_path = config["output_dir"] + f"/accelerate_config.rank{env.rank}.yaml"
    with open(config_path, mode="x", encoding="utf-8") as f:
        print(yaml.dump(ds_config), file=f)
    command = [
        "accelerate",
        "launch",
        "--config_file",
        config_path,
        "run_dpo.py",
    ] + dict2args(config)
    subprocess.run(command)


def main():
    parser = argparse.ArgumentParser("Launch a DPO experiment")
    parser.add_argument("-c", "--config", required=True, help="Configuration YAML")
    parser.add_argument("-d", "--working", required=True, help="Working directory")
    parser.add_argument(
        "--gpus",
        default=None,
        type=int,
        help="Launch through slurm using the given number of GPUs",
    )
    args = parser.parse_args()

    os.makedirs(args.working, exist_ok=True)
    if os.listdir(args.working):
        print("ERROR: Working directory is not empty.", file=sys.stderr)
        sys.exit(-1)

    folder = args.working + "/submitit"
    if args.gpus is None:  # Local
        executor = submitit.LocalExecutor(folder=folder)
        nodes = 1
    else:  # Slurm
        assert args.gpus % GPUS_PER_NODE == 0
        nodes = args.gpus // GPUS_PER_NODE
        executor = submitit.AutoExecutor(folder=folder)

    executor.update_parameters(
        name="dpo",
        nodes=nodes,
        tasks_per_node=1,
        gpus_per_node=GPUS_PER_NODE,
        slurm_gpus_per_task=GPUS_PER_NODE,
        slurm_cpus_per_gpu=4,
        slurm_mem_per_gpu="100GB",
        timeout_min=60 * 24 * 365,  # One year
    )

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f.read())

    config["output_dir"] = args.working
    job = executor.submit(lambda: dpo_task(nodes, config))
    print(f"Launched job {job.job_id}")
    if args.gpus is None:  # Local
        job.results()


if __name__ == "__main__":
    main()
