import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien
import cv2
from spatialvla.inference import VLAModel
from PIL import Image
import numpy as np
from transforms3d.euler import euler2axangle
# task_name = "google_robot_pick_coke_can"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
checkpoint = "checkpoints/vla_rtx_remix_fm_200k"
model = VLAModel(checkpoint)
robot_type = 'google_robot'
# robot_type = 'widowx_bridge'
rollout_per_env = 5

if robot_type == 'google_robot':
    tasks = [
        'google_robot_pick_coke_can',
        'google_robot_pick_object',
        'google_robot_move_near',
        'google_robot_open_drawer',
        'google_robot_close_drawer',
        'google_robot_place_in_closed_drawer'
    ]
    unnorm_key = 'fractal20220817_data'
    hz = 3
else:
    tasks = [
        'widowx_spoon_on_towel',
        'widowx_carrot_on_plate',
        'widowx_stack_cube',
        'widowx_put_eggplant_in_basket'
    ]
    unnorm_key = 'bridge_oxe'
    max_grp = 2.0
    hz = 5
frames = []
grps = []
for task_name in tasks:
    env = simpler_env.make(task_name)
    if robot_type == "google_robot":
        sticky_gripper_num_repeat = 15
    elif robot_type == "widowx_bridge":
        sticky_gripper_num_repeat = 1
    else:
        raise NotImplementedError("Please specify the robot type")
    for trial in range(rollout_per_env):
        # if 'env' in locals():
        #     print("Closing existing env")
        #     env.close()
        obs, reset_info = env.reset()
        instruction = env.get_language_instruction()
        # instruction = env.get_language_instruction()
        print("Reset info", reset_info)
        print("Instruction", instruction)
        done, truncated = False, False
        chunk_stack = 0
        # --------------[reset]-----------------
        actino_scale = 1.0
        previous_gripper_action = None
        sticky_action_is_on = False
        sticky_gripper_action = 0.0
        gripper_action_repeat = 0
        # --------------------------------------
        while not (done or truncated):
            # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
            # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
            image = get_image_from_maniskill2_obs_dict(env, obs)
            if chunk_stack == 0:
                raw_actions = model.inference_action(unnorm_key, Image.fromarray(image), instruction, hz=hz)
            raw_action = {
                "world_vector": np.array(raw_actions[chunk_stack, :3]),
                "rotation_delta": np.array(raw_actions[chunk_stack, 3:6]),
                "open_gripper": np.array(raw_actions[chunk_stack, 6:7]),  # range [0, 1]; 1 = open; 0 = close
            }
            # process raw_action to obtain the action to be sent to the maniskill2 environment
            action = {}
            action["world_vector"] = raw_action["world_vector"] * actino_scale
            action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            roll, pitch, yaw = action_rotation_delta
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = action_rotation_ax * action_rotation_angle
            action["rot_axangle"] = action_rotation_axangle * actino_scale
            if robot_type == "google_robot":
                current_gripper_action = raw_action["open_gripper"]
                if previous_gripper_action is None:
                    relative_gripper_action = np.array([0])
                else:
                    relative_gripper_action = (
                        previous_gripper_action - current_gripper_action
                    )  # google robot 1 = close; -1 = open
                previous_gripper_action = current_gripper_action
                if np.abs(relative_gripper_action) > 0.5 and sticky_action_is_on is False:
                    sticky_action_is_on = True
                    sticky_gripper_action = relative_gripper_action
                if sticky_action_is_on:
                    gripper_action_repeat += 1
                    relative_gripper_action = sticky_gripper_action
                if gripper_action_repeat == sticky_gripper_num_repeat:
                    sticky_action_is_on = False
                    gripper_action_repeat = 0
                    sticky_gripper_action = 0.0
                action["gripper"] = relative_gripper_action
            elif robot_type == "widowx_bridge":
                action["gripper"] = (
                    2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
                )  # binarize gripper action to 1 (open) and -1 (close)
                # self.gripper_is_closed = (action['gripper'] < 0.0)
            action["terminate_episode"] = np.array([0.0])
            chunk_stack += 1
            if chunk_stack == 8:
                chunk_stack = 0
            # grp = action[-1]
            # grps.append(grp)
            obs, reward, done, truncated, info = env.step(np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]))
            frames.append(image)
        episode_stats = info.get('episode_stats', {})
        print("Episode stats", episode_stats)
output_filename = f"rollouts/{checkpoint.split('/')[-1]}.mp4"  # You can change the format to .mp4 if you like
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20  # Define the frames per second of the video
frame_shape = frames[0].shape
height, width, channels = frame_shape
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
# Define text properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (255, 255, 255)  # White text
thickness = 2
position = (10, 50)  # Position of the text in the frame (x, y)
for frame in frames:
    # Ensure the frame is in the correct color format (BGR for OpenCV)
    # print(frame.shape)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # text = str(grp)
    # cv2.putText(frame_bgr, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    out.write(frame_bgr)
out.release()