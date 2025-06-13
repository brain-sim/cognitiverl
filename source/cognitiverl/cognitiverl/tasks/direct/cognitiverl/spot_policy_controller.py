import io

import torch
from isaaclab.utils.math import matrix_from_quat


class SpotPolicyController:
    """
    Minimal Spot policy wrapper. Loads a TorchScript policy and returns actions given observation and command.
    Now fully vectorized: all methods accept and return torch tensors for n environments.
    """

    def __init__(self, policy_file_path: str):
        with open(policy_file_path, "rb") as f:
            self.policy = torch.jit.load(io.BytesIO(f.read()))

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Run the policy on the given batch of observations.
        Args:
            observation (torch.Tensor): [num_envs, obs_dim] observation input.
        Returns:
            torch.Tensor: [num_envs, action_dim] policy output (actions).
        """
        # Ensure policy is on the same device as observation
        if next(self.policy.parameters()).device != observation.device:
            self.policy = self.policy.to(observation.device)
        with torch.no_grad():
            obs_tensor = observation.float()
            action = self.policy(obs_tensor).detach()
        return action

    def compute_observation(
        self,
        lin_vel_I,
        ang_vel_I,
        q_IB,
        command,
        previous_action,
        default_pos,
        joint_pos,
        joint_vel,
    ) -> torch.Tensor:
        """
        Compute the observation vector for the policy for all environments (vectorized).
        Args:
            lin_vel_I: [num_envs, 3] torch.Tensor
            ang_vel_I: [num_envs, 3] torch.Tensor
            q_IB: [num_envs, 4] torch.Tensor (quaternion)
            command: [num_envs, 3] torch.Tensor
            previous_action: [num_envs, 12] torch.Tensor
            default_pos: [num_envs, 12] torch.Tensor
            joint_pos: [num_envs, 12] torch.Tensor
            joint_vel: [num_envs, 12] torch.Tensor
        Returns:
            obs: [num_envs, 48] torch.Tensor
        """
        # Rotation matrices: [num_envs, 3, 3]
        R_IB = matrix_from_quat(q_IB)  # [num_envs, 3, 3]
        R_BI = R_IB.transpose(1, 2)  # [num_envs, 3, 3]
        lin_vel_b = torch.bmm(R_BI, lin_vel_I.unsqueeze(-1)).squeeze(
            -1
        )  # [num_envs, 3]
        ang_vel_b = torch.bmm(R_BI, ang_vel_I.unsqueeze(-1)).squeeze(
            -1
        )  # [num_envs, 3]
        gravity_vec = torch.tensor(
            [0.0, 0.0, -1], device=lin_vel_I.device, dtype=lin_vel_I.dtype
        ).reshape(1, 3, 1)
        gravity_b = torch.bmm(
            R_BI, gravity_vec.expand(lin_vel_I.shape[0], 3, 1)
        ).squeeze(-1)
        obs = torch.zeros(
            (lin_vel_I.shape[0], 48), device=lin_vel_I.device, dtype=lin_vel_I.dtype
        )
        obs[:, 0:3] = lin_vel_b
        obs[:, 3:6] = ang_vel_b
        obs[:, 6:9] = gravity_b
        obs[:, 9:12] = command
        obs[:, 12:24] = joint_pos - default_pos
        obs[:, 24:36] = joint_vel
        obs[:, 36:48] = previous_action
        return obs

    def compute_command(self, goal, state=None) -> torch.Tensor:
        """
        Compute the command vector (e.g., navigation command) for the policy.
        Args:
            goal: The target or goal (type as needed).
            state: Optionally, the current state (type as needed).
        Returns:
            torch.Tensor: The command vector for the policy.
        """
        # TODO: Implement this method for your environment
        raise NotImplementedError(
            "Implement compute_command for your Spot environment."
        )

    def get_action(
        self,
        lin_vel_I,
        ang_vel_I,
        q_IB,
        command,
        previous_action,
        default_pos,
        joint_pos,
        joint_vel,
    ) -> torch.Tensor:
        """
        Compute the observation from the robot state and command, then run the policy to get the action (vectorized).
        All arguments are batched torch tensors.
        Returns:
            actions: [num_envs, action_dim] torch.Tensor
        """
        obs = self.compute_observation(
            lin_vel_I,
            ang_vel_I,
            q_IB,
            command,
            previous_action,
            default_pos,
            joint_pos,
            joint_vel,
        )
        actions = self(obs)
        return actions


if __name__ == "__main__":
    # Simple test for SpotPolicyController
    import numpy as np

    class DummyRobot:
        def get_linear_velocity(self):
            return np.array([1.0, 0.0, 0.0])

        def get_angular_velocity(self):
            return np.array([0.0, 0.0, 0.1])

        def get_world_pose(self):
            return np.array([0.0, 0.0, 0.5]), np.array(
                [1.0, 0.0, 0.0, 0.0]
            )  # pos, quat

        def get_joint_positions(self):
            return np.ones(12)

        def get_joint_velocities(self):
            return np.zeros(12)

    # Dummy data
    dummy_policy_path = "/home/user/cognitiverl/source/cognitiverl/cognitiverl/tasks/direct/cognitiverl/spot_policy.pt"  # Absolute path
    dummy_command = np.array([0.5, 0.0, 0.1])
    dummy_previous_action = np.zeros(12)
    dummy_default_pos = np.ones(12)
    robot = DummyRobot()

    # Test compute_observation and policy call
    try:
        controller = SpotPolicyController(dummy_policy_path)
        obs = controller.compute_observation(
            torch.tensor(robot.get_linear_velocity()),
            torch.tensor(robot.get_angular_velocity()),
            torch.tensor(robot.get_world_pose()[1]),
            torch.tensor(dummy_command),
            torch.tensor(dummy_previous_action),
            torch.tensor(dummy_default_pos),
            torch.tensor(robot.get_joint_positions()),
            torch.tensor(robot.get_joint_velocities()),
        )
        print("Observation vector:", obs)
        print("Observation vector shape:", obs.shape)
        # Now use the computed observation as input to the policy
        action = controller(obs)
        print("Policy output:", action)
    except Exception as e:
        print("[Test] compute_observation or policy call failed:", e)
