import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='DQN')

    # Basic Arguments
    parser.add_argument('--seed', type=int, default=1122,
                        help='Random seed')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--training-id', type=str, default='0',
                        help='ID for training instance')

    # Training Arguments
    parser.add_argument('--max-frames', type=int, default=20000000,
                        help='Number of frames to train')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Maximum memory buffer size')
    parser.add_argument('--update-target', type=int, default=1000,
                        help='Interval of target network update')
    parser.add_argument('--train-freq', type=int, default=1,
                        help='Number of steps between optimization step')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='Anticipatory Parameter for NFSP')
    parser.add_argument('--rl-start', type=int, default=10000,
                        help='How many steps of the model to collect transitions for before RL starts')
    parser.add_argument('--sl-start', type=int, default=1000,
                        help='How many steps of the model to collect transitions for before SL starts')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Number of environments for parallel sampling (needs to be >1 when specified)')

    # Algorithm Arguments
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension of networks')
    parser.add_argument('--dueling', action='store_true',
                        help='Enable Dueling Network')
    parser.add_argument('--multi-step', type=int, default=1,
                        help='N-Step Learning')
    parser.add_argument('--cce', action='store_true',
                        help='Use Coarse Correlated Equilibrium rather than Nash.')

    # Environment Arguments
    parser.add_argument('--env', type=str, default='LaserTag-small2-v0',
                        help='Environment Name')
    parser.add_argument('--ram', action='store_true', default=False,
                        help='Use RAM observation rather than image')
    parser.add_argument('--negative', action='store_true', default=False,
                        help='Give negative(-1) reward for not done.')
    parser.add_argument('--against-baseline', action='store_true', default=False,
                        help='Train the agent against baseline of env.')

    # Evaluation Arguments
    parser.add_argument('--load-model', type=str, default='0',
                        help='Pretrained model name to load (state dict)')
    parser.add_argument('--save-model', type=str, default='0',
                        help='Pretrained model name to save (state dict)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate only')
    parser.add_argument('--render', action='store_true',
                        help='Render evaluation agent')
    parser.add_argument('--evaluation-interval', type=int, default=10000,
                        help='Frames for evaluation interval')

    # Optimization Arguments
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max-tag-interval', type=int, default=10000,
                        help='Maximum length of interval between tagging twice to prevent from non-action, \
                            needs to be larger or (better) equal than maximal episode length of the game!')
    parser.add_argument('--eps-start', type=float, default=1.0,
                        help='Start value of epsilon')
    parser.add_argument('--eps-final', type=float, default=0.01,
                        help='Final value of epsilon')
    parser.add_argument('--eps-decay', type=int, default=30000,
                        help='Adjustment parameter for epsilon')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")  # specify device

    return args
