import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from PIL import Image
import wandb
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# Append current directory so that interpreter can find experiments.robot
# sys.path.append("../..")
from spatialvla.simulation.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from spatialvla.simulation.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from spatialvla.inference import VLAModel


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family = "openvla"
    checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 5                   # Number of rollouts per task
    action_len: int = 8
    num_denoise_steps = 20

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "SpatialVLA-LIBERO"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "jellyho_"          # Name of entity to log under

    seed: int = 48                              # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:  
    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_name

    # Load model
    model = VLAModel(cfg.checkpoint)
    print(model.model.device)
    # model.model.action_head.scheduler = DDIMScheduler(
    #             num_train_timesteps=20,
    #             beta_start=0.0001,
    #             beta_end=0.02,
    #             beta_schedule='squaredcos_cap_v2',
    #             clip_sample=True,
    #             clip_sample_range=5.0,
    #             prediction_type='epsilon',
    #             set_alpha_to_one=True
    #         )
    # model.model.action_head.diffusion_steps = 20
    

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_name}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_name}")
    log_file.write(f"Task suite: {cfg.task_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    action_counter = 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        print(initial_states)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        try:
            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
                print(f"\nTask: {task_description}")
                log_file.write(f"\nTask: {task_description}\n")

                # Reset environment
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])
                action_counter = 0
                # Setup
                t = 0
                replay_images = []
                if cfg.task_name == "libero_spatial":
                    max_steps = 220  # longest training demo has 193 steps
                elif cfg.task_name == "libero_object":
                    max_steps = 280  # longest training demo has 254 steps
                elif cfg.task_name == "libero_goal":
                    max_steps = 300  # longest training demo has 270 steps
                elif cfg.task_name == "libero_10":
                    max_steps = 520  # longest training demo has 505 steps
                elif cfg.task_name == "libero_90":
                    max_steps = 400  # longest training demo has 373 steps

                print(f"Starting episode {task_episodes+1}...")
                log_file.write(f"Starting episode {task_episodes+1}...\n")
                while t < max_steps + cfg.num_steps_wait:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Query model to get action
                    if action_counter == 0:
                        action_chunk = model.inference_action(f'{cfg.task_name}_no_noops', Image.fromarray(img), task_description, hz=15)
                    
                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = action_chunk[action_counter]
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    action = invert_gripper_action(action)
                    # print(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                    action_counter += 1
                    if action_counter == cfg.action_len:
                        action_counter = 0

                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode
                save_rollout_video(
                    replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
                )
                replay_images.clear()

                # Log current results
                print(f"Success: {done}")
                print(f"# episodes completed so far: {total_episodes}")
                print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                log_file.write(f"Success: {done}\n")
                log_file.write(f"# episodes completed so far: {total_episodes}\n")
                log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
                log_file.flush()
        finally:
                # Ensure the environment is properly closed
                if hasattr(env, 'close'):
                    env.close()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )
        

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()