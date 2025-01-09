import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien
import cv2
from spatialvla.inference import VLAModel
from PIL import Image

# task_name = "google_robot_pick_coke_can"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
checkpoint = "checkpoints/rt1_512_16_1gpu"

model = VLAModel(checkpoint)
rollout_per_env = 2
tasks = [
    'google_robot_pick_coke_can',
    'google_robot_pick_object',
    'google_robot_move_near',
    'google_robot_open_drawer',
    'google_robot_close_drawer',
    'google_robot_place_in_closed_drawer'
]
unnorm_key = 'fractal20220817_data'
# tasks = [
#     'widowx_spoon_on_towel',
#     'widowx_carrot_on_plate',
#     'widowx_stack_cube',
#     'widowx_put_eggplant_in_basket'
# ]
# unnorm_key = 'bridge_oxe'
frames = []
for task_name in tasks:
    for trial in range(rollout_per_env):
        if 'env' in locals():
            print("Closing existing env")
            env.close()
            del env
        env = simpler_env.make(task_name)
        obs, reset_info = env.reset()
        instruction = env.get_language_instruction()
        print("Reset info", reset_info)
        print("Instruction", instruction)
        done, truncated = False, False
        chunk_stack = 0
        while not (done or truncated):
            # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
            # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
            image = get_image_from_maniskill2_obs_dict(env, obs)
            
            if chunk_stack == 0:
                actions = model.inference_action(unnorm_key, Image.fromarray(image), instruction)
            action = actions[chunk_stack]
            chunk_stack += 1
            if chunk_stack == 8:
                chunk_stack = 0
            # print(action)
            action[-1] = action[-1] * 2.0 - 1.0
            obs, reward, done, truncated, info = env.step(action)
            frames.append(image)

        episode_stats = info.get('episode_stats', {})
        print("Episode stats", episode_stats)

output_filename = f"rollouts/{checkpoint.split('/')[-1]}.mp4"  # You can change the format to .mp4 if you like
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10  # Define the frames per second of the video
frame_shape = frames[0].shape
height, width, channels = frame_shape
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
for frame in frames:
    # Ensure the frame is in the correct color format (BGR for OpenCV)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)
out.release()