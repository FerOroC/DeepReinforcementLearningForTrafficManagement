#This is needed for the multi agent environment class definition
from gym.spaces.box import Box
from gym.spaces import Tuple
import numpy as np

from flow.core.rewards import average_velocity
from flow.envs.multiagent.base import MultiEnv


additional_env_params = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 5,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 5,
    # whether we use an obs space that contains adjacent lane info or just the lead obs
}

class SharedMultiAgentBristolEnv(MultiEnv):

    @property
    def observation_space(self):
        observation_box=Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(3,),
                dtype=np.float32
            )
        return observation_box
        
    @property
    def action_space(self):
        action_box=Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),
            dtype=np.float32
        )
        return action_box

    def _apply_rl_actions(self, rl_actions):
        # in the warmup steps, rl_actions is None
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                accel = actions[0]
                self.k.vehicle.apply_acceleration(rl_id, accel)

    def get_state(self):
        """See class definition."""
        veh_info = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            speed = self.k.vehicle.get_speed(rl_id)
            headway = self.k.vehicle.get_headway(rl_id)
            lead_speed = self.k.vehicle.get_speed(self.k.vehicle.get_leader(rl_id))
            if lead_speed == -1001:
                lead_speed = 0
            veh_info.update({rl_id: np.array([speed / 50.0, headway / 1000.0, lead_speed / 50.0])})

        return veh_info

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            if kwargs['fail']:
                # reward is 0 if a collision occurred
                reward = 0
            else:
                # reward high system-level velocities
                cost1 = average_velocity(self, fail=kwargs['fail'])

                # penalize small time headways
                cost2 = 0
                t_min = 1  # smallest acceptable time headway

                # weights for cost1, cost2, and cost3, respectively
                eta1, eta2 = 1.00, 0.10

                reward = max(eta1 * cost1 + eta2 * cost2, 0)

            rewards[rl_id] = reward
        return rewards

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes.
        """
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            # leader
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id:
                self.k.vehicle.set_observed(lead_id)
            # follower
            follow_id = self.k.vehicle.get_follower(rl_id)
            if follow_id:
                self.k.vehicle.set_observed(follow_id)

    def state_util(self, rl_id):
        """Return an array of headway, tailway, leader speed, follower speed.

        Also return a 1 if leader is rl 0 otherwise, a 1 if follower is rl 0 otherwise.
        If there are fewer than MAX_LANES the extra
        entries are filled with -1 to disambiguate from zeros.
        """
        veh = self.k.vehicle
        lane_headways = veh.get_lane_headways(rl_id).copy()
        lane_tailways = veh.get_lane_tailways(rl_id).copy()
        lane_leader_speed = veh.get_lane_leaders_speed(rl_id).copy()
        lane_follower_speed = veh.get_lane_followers_speed(rl_id).copy()
        leader_ids = veh.get_lane_leaders(rl_id).copy()
        follower_ids = veh.get_lane_followers(rl_id).copy()
        rl_ids = self.k.vehicle.get_rl_ids()
        is_leader_rl = [1 if l_id in rl_ids else 0 for l_id in leader_ids]
        is_follow_rl = [1 if f_id in rl_ids else 0 for f_id in follower_ids]
        diff = MAX_LANES - len(is_leader_rl)
        if diff > 0:
            # the minus 1 disambiguates missing cars from missing lanes
            lane_headways += diff * [-1]
            lane_tailways += diff * [-1]
            lane_leader_speed += diff * [-1]
            lane_follower_speed += diff * [-1]
            is_leader_rl += diff * [-1]
            is_follow_rl += diff * [-1]
        lane_headways = np.asarray(lane_headways) / 1000
        lane_tailways = np.asarray(lane_tailways) / 1000
        lane_leader_speed = np.asarray(lane_leader_speed) / 100
        lane_follower_speed = np.asarray(lane_follower_speed) / 100
        return np.concatenate((lane_headways, lane_tailways, lane_leader_speed,
                               lane_follower_speed, is_leader_rl,
                               is_follow_rl))

    def veh_statistics(self, rl_id):
        """Return speed, edge information, and x, y about the vehicle itself."""
        speed = self.k.vehicle.get_speed(rl_id) / 100.0
        lane = (self.k.vehicle.get_lane(rl_id) + 1) / 10.0
        return np.array([speed, lane])
