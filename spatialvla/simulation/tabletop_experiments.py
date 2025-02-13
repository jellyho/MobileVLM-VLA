import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import draccus
import numpy as np
import tqdm

from PIL import Image
import torch

from spatialvla.inference import TwinVLAModel
import tabletop
from tabletop.utils import sample_box_pose, sample_insertion_pose

from spatialvla.simulation.libero_utils import (
    save_rollout_video,
)
from spatialvla.simulation.robot_utils import (
    DATE,
    DATE_TIME,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
@dataclass
class Config:
    checkpoint: Union[str, Path] = ""
    task_name: str = "insertion"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    unnorm_key: str = "tabletop_bimanual"
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 5                   # Number of rollouts per task
    action_len: int = 8

    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    seed: int = 48  

@draccus.wrap()
def eval_tabletop(cfg: Config) -> None:
    set_seed_everywhere(cfg.seed)
    unnorm_key = cfg.unnorm_key

    model = TwinVLAModel(cfg.checkpoint)

    ## Logging File ##
    run_id = f"EVAL-{cfg.task_name}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    ##################

    env = tabletop.make_ee_sim_env(cfg.task_name, 'ee_rpy_vel')
    # env = tabletop.env(cfg.task_name, 'ee_rpy_vel')
    highest_rewards = []
    episode_returns = []
    for rollout_id in range(cfg.num_trials_per_task):
        ts = env.reset()
        action_counter = 0
        replay_images = []
        rewards = []
        attns = []
        with torch.inference_mode():
            for t in range(450):
                replay_images.append(env._physics.render(height=240, width=320, camera_id='angle'))
                obs = ts.observation
                img = Image.fromarray(obs['images']['right'])
                secondary_img = Image.fromarray(obs['images']['left'])
                if action_counter == 0:
                    actions, attn = model.inference_action(unnorm_key=unnorm_key, image=img, prompt=env.task.instruction, secondary_image=secondary_img, state=None, output_attn=True, hz=15)
                    # if len(attn) > 0:
                    #     attn = torch.cat(list(attn), axis=0)
                    #     # print(attn.shape)
                    #     attns.append(attn)

                action = actions[action_counter]

                # print(action)
                # if action[6] <= 0.3:
                #     action[6] = 0
                # if action[-1] <= 0.3:
                #     action[-1] = 0
                ts = env.step(action)

                rewards.append(ts.reward)
                action_counter += 1
                if action_counter == cfg.action_len:
                    action_counter = 0
                if ts.reward == env.task.max_reward:
                    break

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        env_max_reward = env.task.max_reward
        
        # attns_weights = torch.stack(attns, axis=0)
        # print(attns_weights.shape)
        save_rollout_video(replay_images, rollout_id, success=episode_highest_reward==env_max_reward, task_description=cfg.task_name, folder=f'{cfg.task_name}/{cfg.run_id_note}')
        os.makedirs(f'./rollouts/{DATE}/{cfg.task_name}/{cfg.run_id_note}/attns', exist_ok=True)
        # if rollout_id < 10:
            # torch.save(attns_weights, f'./rollouts/{DATE}/{cfg.task_name}/{cfg.run_id_note}/attns/attns_weights_{rollout_id}.pt')
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')
        replay_images.clear()

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / cfg.num_trials_per_task
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{cfg.num_trials_per_task} = {more_or_equal_r_rate*100}%\n'
    log_file.write(summary_str)
    log_file.write(repr(episode_returns))
    log_file.write('\n\n')
    log_file.write(repr(highest_rewards))
    log_file.flush()
    log_file.close()


if __name__ == "__main__":
    eval_tabletop()