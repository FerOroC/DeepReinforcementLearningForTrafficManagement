import numpy as np
import re

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces import Tuple

from flow.core import rewards
from flow.envs.base import Env

Additional_Env_Params = {
    "target_velocity": 50, 
    "switch_time": 3.0
}

class LondonTrafficLightEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='traci'):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))


        self.num_traffic_lights = 4
        self.tl_type = env_params.additional_params.get('tl_type')

        super().__init__(env_params, sim_params, network, simulator)

        # Saving env variables for plotting
        self.steps = env_params.horizon
        self.obs_var_labels = {
            'edges': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'velocities': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'positions': np.zeros((self.steps, self.k.vehicle.num_vehicles))
        }

        # Keeps track of the last time the traffic lights in an intersection
        # were allowed to change (the last time the lights were allowed to
        # change from a red-green state to a red-yellow state.)
        self.last_change = np.zeros((self.rows * self.cols, 1))
        # Keeps track of the direction of the intersection (the direction that
        # is currently being allowed to flow. 0 indicates flow from top to
        # bottom, and 1 indicates flow from left to right.)
        self.direction = np.zeros((self.rows * self.cols, 1))
        # Value of 1 indicates that the intersection is in a red-yellow state.
        # value 0 indicates that the intersection is in a red-green state.
        self.currently_yellow = np.zeros((self.rows * self.cols, 1))

        # when this hits min_switch_time we change from yellow to red
        # the second column indicates the direction that is currently being
        # allowed to flow. 0 is flowing top to bottom, 1 is left to right
        # For third column, 0 signifies yellow and 1 green or red
        self.min_switch_time = env_params.additional_params["switch_time"]

        if self.tl_type != "actuated":
            for i in range(self.rows * self.cols):
                self.k.traffic_light.set_state(
                    node_id='center' + str(i), state="GrGr")
                self.currently_yellow[i] = 0

        # # Additional Information for Plotting
        # self.edge_mapping = {"top": [], "bot": [], "right": [], "left": []}
        # for i, veh_id in enumerate(self.k.vehicle.get_ids()):
        #     edge = self.k.vehicle.get_edge(veh_id)
        #     for key in self.edge_mapping:
        #         if key in edge:
        #             self.edge_mapping[key].append(i)
        #             break

        # check whether the action space is meant to be discrete or continuous
        self.discrete = env_params.additional_params.get("discrete", False)

    @property
    def action_space(self):
        if self.discrete:
                return Discrete(2 ** self.num_traffic_lights)
            else:
                return Box(
                    low=0,
                    high=1,
                    shape=(self.num_traffic_lights,),
                    dtype=np.float32)

    @property
    def observation_space(self):
        speed = Box(
                low=0,
                high=1,
                shape=(self.initial_vehicles.num_vehicles,),
                dtype=np.float32)
        dist_to_intersec = Box(
                low=0.,
                high=np.inf,
                shape=(self.initial_vehicles.num_vehicles,),
                dtype=np.float32)
        edge_num = Box(
                low=0.,
                high=1,
                shape=(self.initial_vehicles.num_vehicles,),
                dtype=np.float32)
        traffic_lights = Box(
                low=0.,
                high=1,
                shape=(3 * self.rows * self.cols,),
                dtype=np.float32)
        return Tuple((speed, dist_to_intersec, edge_num, traffic_lights))

    def get_state(self):
    # compute the normalizers
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"],
                       grid_array["long_length"],
                       grid_array["inner_length"])

        # get the state arrays
        speeds = [
            self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
            for veh_id in self.k.vehicle.get_ids()
        ]
        dist_to_intersec = [
            self.get_distance_to_intersection(veh_id) / max_dist
            for veh_id in self.k.vehicle.get_ids()
        ]
        edges = [
            self._convert_edge(self.k.vehicle.get_edge(veh_id)) /
            (self.k.network.network.num_edges - 1)
            for veh_id in self.k.vehicle.get_ids()
        ]

        state = [
            speeds, dist_to_intersec, edges,
            self.last_change.flatten().tolist(),
            self.direction.flatten().tolist(),
            self.currently_yellow.flatten().tolist()
        ]
        return np.array(state)



    def compute_reward(self, rl_actions, **kwargs):
        return - rewards.min_delay_unscaled(self) - rewards.boolean_action_penalty(rl_actions >= 0.5, gain=1.0)

    