import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien
import cv2
from spatialvla.inference import VLAModel
from PIL import Image

# task_name = "google_robot_pick_coke_can"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
checkpoint = "checkpoints/rt1_512_16_4gpu"

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
max_grp = -2.0
unnorm_key = 'fractal20220817_data'
tasks = [
    'widowx_spoon_on_towel',
    'widowx_carrot_on_plate',
    'widowx_stack_cube',
    'widowx_put_eggplant_in_basket'
]
unnorm_key = 'bridge_oxe'
max_grp = 2.0

frames = []
grps = []
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
            grp = action[-1]
            # if grp < -0.1:
            #     grp = -0.1
            # elif grp > 0.1:
            #     grp = 0.1
            # else:
            #     grp = 0
            # action[-1] = grp * max_grp + max_grp / 2
            action[-1] = grp
            grps.append(grp)
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

# Define text properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (255, 255, 255)  # White text
thickness = 2
position = (10, 50)  # Position of the text in the frame (x, y)

for grp, frame in zip(grps, frames):
    # Ensure the frame is in the correct color format (BGR for OpenCV)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    text = str(grp)
    cv2.putText(frame_bgr, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    out.write(frame_bgr)
out.release()