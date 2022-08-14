import re
from flow.envs.base import Env
from flow.core.rewards import average_velocity

from gym.spaces.box import Box
import numpy as np
    
def avg_delay_specified_vehicles(env, veh_ids):
    """Calculate the average delay for a set of vehicles in the system.

    Parameters
    ----------
    env: flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    veh_ids: a list of the ids of the vehicles, for which we are calculating
        average delay
    Returns
    -------
    float
        average delay
    """
    sum = 0
    for edge in env.k.network.get_edge_list():
        for veh_id in env.k.vehicle.get_ids_by_edge(edge):
            v_top = env.k.network.speed_limit(edge)
            sum += (v_top - env.k.vehicle.get_speed(veh_id)) / v_top
    time_step = env.sim_step
    try:
        cost = time_step * sum
        return cost / len(veh_ids)
    except ZeroDivisionError:
        return 0


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 1,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 1,
    # lane change duration for autonomous vehicles, in s. Autonomous vehicles
    # reject new lane changing commands for this duration after successfully
    # changing lanes.
    "lane_change_duration": 5,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 10,
    # specifies whether vehicles are to be sorted by position during a
    # simulation step. If set to True, the environment parameter
    # self.sorted_ids will return a list of all vehicles sorted in accordance
    # with the environment
    'sort_vehicles': False
}

class RouterEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        # variables used to sort vehicles by their initial position plus
        # distance traveled
        self.prev_pos = dict()
        self.absolute_position = dict()
        self.count_destination = 0

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lower_bound = [-abs(max_decel), 0] * self.initial_vehicles.num_rl_vehicles
        upper_bound = [max_accel, 3] * self.initial_vehicles.num_rl_vehicles

        action_Box=Box(
            low=np.array(lower_bound),
            high=np.array(upper_bound),
            dtype=np.float32
        )
        return action_Box

    @property
    def observation_space(self):
        """See class definition. Chjange below to set bounds if not"""
        return Box(
            low=float('-inf'),
            high=float('inf'),
            shape=(3 * self.initial_vehicles.num_vehicles, ),
            dtype=np.float32)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # compute the system-level performance of vehicles from a velocity
        # perspective

        destination_reward=0

        cost1 = avg_delay_specified_vehicles(self, 'rl')

        cost2 = average_velocity(self, fail=kwargs['fail'])

        for veh_id in self.k.vehicle.get_rl_ids():
            if self.k.vehicle.get_edge(veh_id)=="edge_NS_11":
                destination_reward=5000
                self.count_destination+=1

        reward = cost1 + cost2 + destination_reward

        return reward

    def get_state(self):
        """See class definition."""
        # normalizers
        max_speed = self.k.network.max_speed()
        length = self.k.network.length()
        max_lanes = max(
            self.k.network.num_lanes(edge)
            for edge in self.k.network.get_edge_list())

        speed = [self.k.vehicle.get_speed(veh_id) / max_speed
                 for veh_id in self.sorted_ids]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / length
               for veh_id in self.sorted_ids]
        lane = [self.k.vehicle.get_lane(veh_id) / max_lanes
                for veh_id in self.sorted_ids]

        return np.array(speed + pos + lane)

    def _apply_rl_actions(self, rl_actions):

        acceleration=rl_actions[::2]
        directions=rl_actions[1::2]

        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()
        ]
        counter=0
        for i in range(len(sorted_rl_ids)):
            veh_id=sorted_rl_ids[i]
            sampled_action=int(directions[i])
            if sampled_action==3:
                sampled_action=2
            veh_edge = self.k.vehicle.get_edge(veh_id)
            veh_route = self.k.vehicle.get_route(veh_id)
            veh_next_edge = self.k.network.next_edge(veh_edge,
                                                    self.k.vehicle.get_lane(veh_id))
            not_an_edge = ":"
            no_next = 0

            if len(veh_next_edge) == no_next:
                next_route = None
            elif veh_route[-1] == veh_edge:
                if len(veh_next_edge)==3:
                    veh_action=sampled_action
                elif len(veh_next_edge)==2:
                    if sampled_action==2:
                        veh_action=0
                    else:
                        veh_action=sampled_action
                elif len(veh_next_edge)==1:
                    veh_action=0
                while veh_next_edge[0][0][0] == not_an_edge:
                    veh_next_edge = self.k.network.next_edge(
                    veh_next_edge[veh_action][0],
                    veh_next_edge[veh_action][1])
                next_route = [veh_edge, veh_next_edge[0][0]]
            else:
                next_route = None
            self.k.vehicle.choose_routes(veh_id, next_route)
        self.k.vehicle.apply_acceleration(sorted_rl_ids,acceleration)

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)
                
        for veh_id in self.k.vehicle.get_rl_ids():
            self._reroute_if_final_edge(veh_id)        
    
    def _reroute_if_final_edge(self, veh_id):
        """Reroute vehicle associated with veh_id.

        Checks if an edge is the final edge. If it is return the route it
        should start off at.
        """
        edge = self.k.vehicle.get_edge(veh_id)
        if edge == "":
            return
        if edge[0] == ":":  # center edge
            return
        pattern = re.compile(r"[a-zA-Z]+")
        edge_type = pattern.match(edge).group()
        edge = edge.split(edge_type)[1].split('_')

        # find the route that we're going to place the vehicle on if we are
        # going to remove it
        route_id = None
        if edge_type == 'destination':
            route_id = "edge_WE_1"

        if route_id is not None:
            type_id = self.k.vehicle.get_type(veh_id)
            lane_index = self.k.vehicle.get_lane(veh_id)
            # remove the vehicle
            self.k.vehicle.remove(veh_id)
            # reintroduce it at the start of the network
            print("000")
            self.k.vehicle.add(
                veh_id=veh_id,
                edge=route_id,
                type_id=str(type_id),
                lane=str(lane_index),
                pos="0",
                speed="max")

    @property
    def sorted_ids(self):
        """Sort the vehicle ids of vehicles in the network by position.

        This environment does this by sorting vehicles by their absolute
        position, defined as their initial position plus distance traveled.

        Returns
        -------
        list of str
            a list of all vehicle IDs sorted by position
        """
        if self.env_params.additional_params['sort_vehicles']:
            return sorted(self.k.vehicle.get_ids(), key=self._get_abs_position)
        else:
            return self.k.vehicle.get_ids()

    def _get_abs_position(self, veh_id):
        """Return the absolute position of a vehicle."""
        return self.absolute_position.get(veh_id, -1001)

    def reset(self):
        """See parent class.

        This also includes updating the initial absolute position and previous
        position.
        """
        obs = super().reset()

        for veh_id in self.k.vehicle.get_ids():
            self.absolute_position[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
            self.prev_pos[veh_id] = self.k.vehicle.get_x_by_id(veh_id)

        return obs
