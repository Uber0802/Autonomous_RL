"""Define cost functions following GRAPE
https://github.com/aiming-lab/GRAPE/blob/main/Data%20Collection/maniskill2_evaluator.py
"""
import numpy as np


class GrapeCostCalculator:

    def __init__(self, reset_info, alpha=[1, 1, 1], beta=[0.01, 0.01, 0.01],
                 threshold=[[20,40,2], [0,5,2], [0,10,2]], stage_num=3):
        assert len(alpha) == len(beta) == len(threshold) == stage_num

        self.reset_info = reset_info
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.stage_num = stage_num

        # Initialize self.cost and self.cost_sum_dict
        self.reset()

    def reset(self, reset_info=None):
        self.cost = 0
        self.cost_sum_dict = {
            'col_cost': [0,0,0],
            'grasp_cost': [0,0,0],
            'path_cost': [0,0,0],
        }
        if reset_info is not None:
            self.reset_info = reset_info

    @staticmethod
    def stage1_subgoal_constraint1(end_effector, keypoints):  #
        """Align end-effector with the carrot's center."""
        carrot_center = keypoints[0]  # Assuming keypoint[0] is carrot center
        path_cost = np.linalg.norm(end_effector - carrot_center)
        return path_cost

    @staticmethod
    def stage1_collision_constraint1(end_effector, keypoints):
        """Ensure the end-effector approaches from above."""
        carrot_center = keypoints[0]
        # Check if the z-coordinate of the end-effector is higher than the carrot's z-coordinate
        collision_cost = 0 if end_effector[2] > carrot_center[2] else 1  # Penalize if below the carrot
        return collision_cost

    @staticmethod
    def stage1_grasp_constraint(grasp_status, is_src_obj_grasped):
        """Grasp the carrot."""
        grasp_cost = 0 if is_src_obj_grasped else 1  # Grasp cost is incurred when the carrot is grasped
        return grasp_cost

    ### Stage 2: Move carrot to plate
    # The carrot must stay grasped and avoid collisions.
    @staticmethod
    def stage2_grasp_constraint(grasp_status, is_src_obj_grasped, src_on_target):
        """Ensure the carrot remains grasped during the move."""
        grasp_cost = 0 if is_src_obj_grasped else 1  # Carrot must remain grasped
        return grasp_cost

    @staticmethod
    def stage2_collision_constraint(end_effector, keypoints):
        """Ensure the carrot is aligned above the plate."""
        carrot_center = keypoints[0]
        plate_center = keypoints[1]  # Assuming keypoint[1] is the plate center
        collision_cost = np.linalg.norm(carrot_center[:2] - plate_center[:2])  # Only consider x and y axes
        return collision_cost

    ### Stage 3: Drop carrot on plate
    # Ensure the carrot is placed on the plate and avoid collision.
    @staticmethod
    def stage3_path_constraint(end_effector, keypoints):
        """Place the carrot on the plate."""
        carrot_center = keypoints[0]
        plate_center = keypoints[1]
        path_cost = np.linalg.norm(carrot_center - plate_center)  # Ensure carrot is on the plate center
        return path_cost

    @staticmethod
    def stage3_collision_constraint(end_effector, keypoints):
        """Ensure end-effector moves away after placing the carrot."""
        carrot_center = keypoints[0]
        # Check if the end-effector moves above and away after placing
        collision_cost = 0 if end_effector[2] > carrot_center[2] else 1
        return collision_cost

    def cal_cost(self, end_effector, keypoints, stage, info):

        cost = {}
        grasp_status = info['is_src_obj_grasped']
        is_src_obj_grasped = info['is_src_obj_grasped']
        src_on_target = info['src_on_target']

        if stage == 1:
            cost['path_cost'] = self.stage1_subgoal_constraint1(
                end_effector, keypoints)
            cost['col_cost'] = self.stage1_collision_constraint1(
                end_effector, keypoints)
            cost['grasp_cost'] = self.stage1_grasp_constraint(
                grasp_status, is_src_obj_grasped)
        elif stage == 2:
            cost['grasp_cost'] = self.stage2_grasp_constraint(
                grasp_status, is_src_obj_grasped, src_on_target)
            cost['col_cost'] = self.stage2_collision_constraint(
                end_effector, keypoints)
        elif stage == 3:
            cost['path_cost'] = self.stage3_path_constraint(
                end_effector, keypoints)
            cost['col_cost'] = self.stage3_collision_constraint(
                end_effector, keypoints)

        cost_sum = 0
        for k, v in cost.items():
            cost_sum += v

        return cost_sum, cost, stage

    def external_score_option1(self):

        matrix = np.array([
            self.cost_sum_dict['col_cost'],
            self.cost_sum_dict['grasp_cost'],
            self.cost_sum_dict['path_cost']
        ])

        transposed_matrix = matrix.T
        cost_matrix = transposed_matrix
        
        ex_score = 1

        for stage in range(self.stage_num):
            stage_score = 1
            panel = 0

            for cons in range(self.stage_num):
                panel += (
                    self.alpha[cons] *
                    max(0, cost_matrix[stage][cons] - self.threshold[stage][cons])
                )
            stage_score -= panel

            ex_score *= stage_score

        return ex_score

    def external_score_option2(self):
        matrix = np.array([
            self.cost_sum_dict['col_cost'],
            self.cost_sum_dict['grasp_cost'],
            self.cost_sum_dict['path_cost']
        ])

        transposed_matrix = matrix.T
        cost_matrix = transposed_matrix

        ex_score = 1
        stage_score=[]

        for stage in range (self.stage_num):
            stage_score = 0
            panel = 0
            
            for cons in range(self.stage_num):
                panel -= (
                    self.beta[cons] *
                    max(0, cost_matrix[stage][cons] - self.threshold[stage][cons])
                )
            stage_score = np.exp(panel)

            ex_score *= stage_score

        return ex_score    

    def _env_step_callback(self, env, obs, reward, done, truncated, info, stepped_action):
        """A callback function to calculate the cost of the current step after
        calling env.step().

        Args:
            env: Gym environment object.
            obs: A dictionary containing the observation data.
            reward: A float value representing the reward.
            done: A boolean value indicating if the episode is done.
            truncated: A boolean value indicating if the episode is truncated.
            info: A dictionary containing additional information.
            stepped_action: A np.array indicating the action taken by the agent.
        """
        if "widowx" in env.unwrapped.robot_uid:
            obs_pose = (
                obs['agent']['controller']['arm']['target_pose'][:3]
            )
            source_pose = (
                self.reset_info['episode_source_obj_init_pose_wrt_robot_base']
            )
            tar_pose = (
                self.reset_info['episode_target_obj_init_pose_wrt_robot_base'] 
            )

            if info["moved_correct_obj"] == False:
                stage = 1
            elif (info["moved_correct_obj"] == True and
                info['is_src_obj_grasped'] == True):
                stage = 2
            elif (info['is_src_obj_grasped'] == True and
                info['consecutive_grasp'] == True and
                info["src_on_target"]== False):
                stage = 3
            else:
                stage = 1

            keypoints = [
                np.array(source_pose.p),np.array(tar_pose.p)
            ]

            cost_step, cost_dict, stage = self.cal_cost(
                end_effector=obs_pose, keypoints=keypoints, stage=stage, info=info
            )

            self.cost += cost_step

            for k,v in cost_dict.items():
                self.cost_sum_dict[k][stage-1]+=v
        
    def env_step_callback(self, env, obs, reward, done, truncated, info, stepped_action):
        """A callback function to calculate the cost of the current step after
        calling env.step().

        Args:
            env: Gym environment object.
            obs: A dictionary containing the observation data.
            reward: A float value representing the reward.
            done: A boolean value indicating if the episode is done.
            truncated: A boolean value indicating if the episode is truncated.
            info: A dictionary containing additional information.
            stepped_action: A np.array indicating the action taken by the agent.
        """
        return self._env_step_callback(
            env, obs, reward, done, truncated, info, stepped_action
        )