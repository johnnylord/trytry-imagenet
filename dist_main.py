import sys
import signal
import os
import os.path as osp
import argparse
import yaml
from agent import get_agent_cls

import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    dist.destroy_process_group()
    sys.exit(0)

def dist_training(i, agent_cls, config):
    print("Launch distributed worker:{}".format(i))
    # Make sure sampler and model is initialized to the same state between processes
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Master process information between all processes
    os.environ['MASTER_ADDR'] = config['train']['master_addr']
    os.environ['MASTER_PORT'] = config['train']['master_port']
    world_size=len(config['train']['gpus'])
    dist.init_process_group("nccl", rank=i, world_size=world_size)

    # Start training
    agent = agent_cls(config, rank=i)
    agent.train()
    agent.finalize()

def main(config_path):
    with open(config_path) as f:
        config = yaml.full_load(f)

    agent_cls = get_agent_cls(config['train']['agent'])
    mp.spawn(dist_training, nprocs=len(config['train']['gpus']), args=(agent_cls, config))

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="path to configuration file")

    args = vars(parser.parse_args())
    main(args['config'])
