#!/usr/bin/env python3
import os
import time
import gym
import argparse
from tensorboardX import SummaryWriter

import model, common, experience, utils

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

ENV_NAME = "LunarLanderContinuous-v2" ; 
STOP_REWARD = 180; 
VAL_SCALE = 1

GAMMA = 0.999
REWARD_STEPS = 2
BATCH_SIZE = 256
LEARNING_RATE = 0.0005
ENTROPY_BETA = 0.001
TEST_ITERS = 50_000
MAX_STEPS = 1_000_000
NUM_ENVS = 20
CLIP_GRAD = 1.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable CUDA"
    )
    # parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    save_path = os.path.join("saves", f"a2c-{ENV_NAME}")
    os.makedirs(save_path, exist_ok=True)

    envs = [gym.make(ENV_NAME) for _ in range(NUM_ENVS)]
    test_env = gym.make(ENV_NAME)

    net = model.ModelA2C(
        test_env.observation_space.shape[0],
        test_env.action_space.shape[0],
        val_scale=VAL_SCALE,
    ).to(device)

    agent = model.AgentA2C(net, device=device)
    exp_source = experience.ExperienceSourceFirstLast(
        envs, agent, GAMMA, REWARD_STEPS
    )

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    run_name = f"B{BATCH_SIZE}_M{NUM_ENVS}_N{REWARD_STEPS}_V{VAL_SCALE}_b{ENTROPY_BETA}_L{LEARNING_RATE}"
    writer = SummaryWriter(
        comment=f"-a2c-{ENV_NAME}-" + run_name
    )
    batch = []
    best_reward = None

    with utils.RewardTracker(writer) as tracker:
        with utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                if step_idx > MAX_STEPS:
                    print(f"Training Stopped after {MAX_STEPS}!")
                    break
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    with torch.no_grad():
                        rewards, steps = common.test_net(net, test_env, device=device)
                    print(
                        "Test done is %.2f sec, reward %.2f, steps %d"
                        % (time.time() - ts, rewards, steps)
                    )
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print(
                                "Best reward updated: %.2f -> %.2f"
                                % (best_reward, rewards)
                            )
                            name = "best_%+.2f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net.state_dict(), fname)
                        best_reward = rewards
                        if best_reward > STOP_REWARD:
                            print("Solved!")
                            break

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = common.unpack_batch_a2c(
                    batch, net, GAMMA ** REWARD_STEPS, device
                )
                batch.clear()

                optimizer.zero_grad()
                mu_v, var_v, value_v = net(states_v)
                loss_value = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                adv = vals_ref_v.unsqueeze(-1) - value_v.detach()
                log_prob = adv * common.calc_logprob(mu_v, var_v, actions_v)
                loss_policy = -log_prob.mean()

                # entropy for normal distribution
                neg_entropy = -((torch.log(2 * np.pi * var_v) + 1) / 2).mean()
                loss_entropy = ENTROPY_BETA * neg_entropy

                loss = loss_policy + loss_value + loss_entropy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()

                # if var_v.mean() > 5:
                #     print("variance too large!")
                #     break

                tb_tracker.track("loss_entropy", loss_entropy, step_idx)
                tb_tracker.track("loss_policy", loss_policy, step_idx)
                tb_tracker.track("loss_value", loss_value, step_idx)
                # tb_tracker.track("loss_total", loss, step_idx)
                # tb_tracker.track("mean_var", var_v.mean(), step_idx)

