# Command Instruction

### Single Agent against Baseline

For training DQN against the environment baseline:

1. single process version code:

   `python train_dqn_against_baseline.py  --env SlimeVolley-v0 --hidden-dim 256`

   ```python train_dqn_against_baseline.py  --env Pong-ram-v0 --hidden-dim 32```

   Note: 

   * For `SlimeVolley` env, use `SlimeVolley-v0` for RAM control and `SlimeVolleyNoFrameskip-v0` for image-based (3\*168\*84) control; and it requires the `hidden-dim` to be 256 to learn effective models.
   * For `Pong` env (OpenAI Gym Atari Pong), use `Pong-ram-v0` for RAM control and `Pong-v0` for image-based control; the default `hidden-dim` 32 can solve the RAM version within an hour.

2. multi-process version code (with vectorized environments):

`python train_dqn_against_baseline_mp.py  --env SlimeVolley-v0 --hidden-dim 256 --num-envs 5`

`python train_dqn_against_baseline_mp.py  --env Pong-ram-v0 --hidden-dim 256 --num-envs 2` 

### Two Agents Nash DQN

For two agents zero-sum game with Nash DQN:

`python nash_dqn.py  --env SlimeVolley-v0 --hidden-dim 256 --num_envs 5`

`python nash_dqn.py  --env pong_v1 --ram --hidden-dim 32 --num_envs 2`

Note: 

* `pong_v1` is the `Pong` game from PettingZoo for two agents, need to specify `--ram` for RAM control, otherwise it is image-based control.

### Two Agents Neural Fictitious Self-Play (NFSP)

`python main.py --env SlimeVolley-v0 --hidden-dim 256 --max-frames 20000000`

`python main.py --env SlimeVolleyNoFrameskip-v0 --hidden-dim 512 --max-frames 30000000`

`python main.py --env pong_v1 --ram --max-frames 20000000`

