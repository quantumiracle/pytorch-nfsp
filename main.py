import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import lasertag
import time, os
from tensorboardX import SummaryWriter

from common.utils import create_log_dir, print_args, set_global_seeds
from common.wrappers import wrap_pytorch, make_env
from arguments import get_args
from train import train
from test import test

def main():
    args = get_args()
    print_args(args)

    log_dir = create_log_dir(args)
    if not args.evaluate:
        writer = SummaryWriter(log_dir)
    SEED = 721
    env = make_env(args)  # "LaserTag-small2-v0"   "SlimeVolleyPixel-v0"

    print(env.observation_space, env.action_space)

    set_global_seeds(args.seed)
    env.seed(args.seed)

    if args.evaluate:
        test(env, args)
        env.close()
        return

    train(env, args, writer)

    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
