from flow.envs.base import Env
from flow.core import rewards

from gym.spaces.box import Box
import numpy as np
from numpy.lib.arraypad import _get_edges

def average_velocity(env, fail=False):
    """Encourage proximity to an average velocity.

    This reward function returns the average velocity of all
    vehicles in the system.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    fail : bool, optional
        specifies if any crash or other failure occurred in the system

    Returns
    -------
    float
        reward value
    """
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

    if any(vel < -100) or fail:
        return 0.
    if len(vel) == 0:
        return 0.

    return np.mean(vel)

def penalize_near_standstill(env, thresh=6, gain=0.075):
    """Reward function which penalizes vehicles at a low velocity.
    #For London Network********

    This reward function is used to penalize vehicles below a
    specified threshold. This assists with discouraging RL from
    gamifying a network, which can result in standstill behavior
    or similarly bad, near-zero velocities.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
    thresh : float
        the velocity threshold below which penalties are applied
    gain : float
        multiplicative factor on the action penalty
    """
    veh_ids = env.k.vehicle.get_ids()
    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    penalize = len(vel[vel < thresh])
    penalty = gain * penalize
    return -penalty

def desired_velocity(env, fail=False, edge_list=None):
    r"""Encourage proximity to a desired velocity.

    This function measures the deviation of a system of vehicles from a
    user-specified desired velocity peaking when all vehicles in the ring
    are set to this desired velocity. Moreover, in order to ensure that the
    reward function naturally punishing the early termination of rollouts due
    to collisions or other failures, the function is formulated as a mapping
    :math:`r: \\mathcal{S} \\times \\mathcal{A}
    \\rightarrow \\mathbb{R}_{\\geq 0}`.
    This is done by subtracting the deviation of the system from the
    desired velocity from the peak allowable deviation from the desired
    velocity. Additionally, since the velocity of vehicles are
    unbounded above, the reward is bounded below by zero,
    to ensure nonnegativity.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    fail : bool, optional
        specifies if any crash or other failure occurred in the system
    edge_list : list  of str, optional
        list of edges the reward is computed over. If no edge_list is defined,
        the reward is computed over all edges

    Returns
    -------
    float
        reward value
    """
    if edge_list is None:
        veh_ids = env.k.vehicle.get_ids()
    else:
        veh_ids = env.k.vehicle.get_ids_by_edge(edge_list)

    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    num_vehicles = len(veh_ids)

    if any(vel < -100) or fail or num_vehicles == 0:
        return 0.

    target_vel = env.env_params.additional_params['target_velocity']
    max_cost = np.array([target_vel] * num_vehicles)
    max_cost = np.linalg.norm(max_cost)

    cost = vel - target_vel
    cost = np.linalg.norm(cost)

    # epsilon term (to deal with ZeroDivisionError exceptions)
    eps = np.finfo(np.float32).eps

    return 10*(max(max_cost - cost, 0) / (max_cost + eps))
    
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
            sum += (7.0 - env.k.vehicle.get_speed(veh_id)) / 7.0
    time_step = env.sim_step
    try:
        cost = time_step * sum
        return cost / len(veh_ids)
    except ZeroDivisionError:
        return 0


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 2,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 2,
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

class BusLaneEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        # variables used to sort vehicles by their initial position plus
        # distance traveled
        self.prev_pos = dict()
        self.absolute_position = dict()

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [-abs(max_decel), -1] * self.initial_vehicles.num_rl_vehicles
        ub = [max_accel, 1] * self.initial_vehicles.num_rl_vehicles

        return Box(np.array(lb), np.array(ub), dtype=np.float32)

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


        cost1 = -avg_delay_specified_vehicles(self, 'bus')

        cost2 = average_velocity(self)

        cost3 = penalize_near_standstill(self)

        cost4 = desired_velocity(self)

        # print("avg delay specified vehicles= ", 6*cost1)
        # print("average velocity= ", cost2)
        # print("penalise near standstill= ", cost3)
        # print("desired velocity ", cost4)
        # print("------------------------")

        #Change cost 1 coefficient from 10 to 3 for London grid network, and change gain (0.1->0.05) for standstill penalty as way more vehicles are present and some stop at red lifghts 
        #change to 3
        reward = ((3*cost1) + cost2 + cost3)

        # punish excessive lane changes by reducing the reward by a set value
        # every time an rl car changes lanes (10% of max reward)
        for veh_id in self.k.vehicle.get_rl_ids():
            if self.k.vehicle.get_last_lc(veh_id) == self.time_counter:
                reward -= 10
        return 0.1*reward

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

    def _apply_rl_actions(self, actions):
        """See class definition."""
        acceleration = actions[::2]
        direction = actions[1::2]

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()
        ]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.time_counter <=
             self.env_params.additional_params["lane_change_duration"]
             + self.k.vehicle.get_last_lc(veh_id)
             for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = \
            np.array([0] * sum(non_lane_changing_veh))


        for i in range(len(direction)):
            if direction[i]>0.333:
                direction[i]=1
            if direction[i]<-0.333:
                direction[i]=-1
            if (direction[i]<=0.333 and direction[i]>=-0.333):
                direction[i]=0

        self.k.vehicle.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.k.vehicle.apply_lane_change(sorted_rl_ids, direction=direction)

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)
    
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
