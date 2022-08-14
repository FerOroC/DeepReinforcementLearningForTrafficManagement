"""Contains an experiment class for running simulations."""
from flow.utils.registry import make_create_env
from datetime import datetime
import logging
import time
import numpy as np


class Experiment:
    """
    Class for systematically running simulations in any supported simulator.

    This class acts as a runner for a network and environment. In order to use
    it to run an network and environment in the absence of a method specifying
    the actions of RL agents in the network, type the following:

        >>> from flow.envs import Env
        >>> flow_params = dict(...)  # see the examples in exp_config
        >>> exp = Experiment(flow_params)  # for some experiment configuration
        >>> exp.run(num_runs=1)

    If you wish to specify the actions of RL agents in the network, this may be
    done as follows:

        >>> rl_actions = lambda state: 0  # replace with something appropriate
        >>> exp.run(num_runs=1, rl_actions=rl_actions)

    Finally, if you would like to like to plot and visualize your results, this
    class can generate csv files from emission files produced by sumo. These
    files will contain the speeds, positions, edges, etc... of every vehicle
    in the network at every time step.

    In order to ensure that the simulator constructs an emission file, set the
    ``emission_path`` attribute in ``SimParams`` to some path.

        >>> from flow.core.params import SimParams
        >>> flow_params['sim'] = SimParams(emission_path="./data")

    Once you have included this in your environment, run your Experiment object
    as follows:

        >>> exp.run(num_runs=1, convert_to_csv=True)

    After the experiment is complete, look at the "./data" directory. There
    will be two files, one with the suffix .xml and another with the suffix
    .csv. The latter should be easily interpretable from any csv reader (e.g.
    Excel), and can be parsed using tools such as numpy and pandas.

    Attributes
    ----------
    custom_callables : dict < str, lambda >
        strings and lambda functions corresponding to some information we want
        to extract from the environment. The lambda will be called at each step
        to extract information from the env and it will be stored in a dict
        keyed by the str.
    env : flow.envs.Env
        the environment object the simulator will run
    """

    def __init__(self, flow_params, custom_callables=None):
        """Instantiate the Experiment class.

        Parameters
        ----------
        flow_params : dict
            flow-specific parameters
        custom_callables : dict < str, lambda >
            strings and lambda functions corresponding to some information we
            want to extract from the environment. The lambda will be called at
            each step to extract information from the env and it will be stored
            in a dict keyed by the str.
        """
        self.custom_callables = custom_callables or {}

        # Get the env name and a creator for the environment.
        create_env, _ = make_create_env(flow_params)

        # Create the environment.
        self.env = create_env()

        logging.info(" Starting experiment {} at {}".format(
            self.env.network.name, str(datetime.utcnow())))

        logging.info("Initializing environment.")

    def run(self, num_runs, rl_actions=None, convert_to_csv=False):
        """Run the given network for a set number of runs.

        Parameters
        ----------
        num_runs : int
            number of runs the experiment should perform
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any)
        convert_to_csv : bool
            Specifies whether to convert the emission file created by sumo
            into a csv file

        Returns
        -------
        info_dict : dict < str, Any >
            contains returns, average speed per step
        """
        num_steps = self.env.env_params.horizon

        # raise an error if convert_to_csv is set to True but no emission
        # file will be generated, to avoid getting an error at the end of the
        # simulation
        if convert_to_csv and self.env.sim_params.emission_path is None:
            raise ValueError(
                'The experiment was run with convert_to_csv set '
                'to True, but no emission file will be generated. If you wish '
                'to generate an emission file, you should set the parameter '
                'emission_path in the simulation parameters (SumoParams or '
                'AimsunParams) to the path of the folder where emissions '
                'output should be generated. If you do not wish to generate '
                'emissions, set the convert_to_csv parameter to False.')

        # used to store
        info_dict = {
            "returns": [],
            "velocities": [],
            "outflows": [],
        }
        info_dict.update({
            key: [] for key in self.custom_callables.keys()
        })

        if rl_actions is None:
            def rl_actions(*_):
                return None

        # time profiling information
        t = time.time()
        times = []


        vehicle_ids=[]
        bus_speeds = []
        mean_bus_speed = []
        nonRL_speeds=[]
        mean_nonRL_vehicle_speed=[]
        RL_speeds=[]
        mean_RL_vehicle_speed=[]
        human_speeds=[]
        mean_human_vehicle_speed=[]

        AverageReward=[]
        Times=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        for i in range(num_runs):
            ret = 0
            vel = []

            vel = []
            RL_vel=[]
            bus_vel=[]
            human_vel=[]
            nonRL_vel=[]

            custom_vals = {key: [] for key in self.custom_callables.keys()}
            state = self.env.reset()
            for j in range(num_steps):
                t0 = time.time()
                state, reward, done, _ = self.env.step(rl_actions(state))
                t1 = time.time()
                times.append(1 / (t1 - t0))

                # Compute the velocity speeds and cumulative returns.
                veh_ids = self.env.k.vehicle.get_ids()
                vel.append(np.mean(self.env.k.vehicle.get_speed(veh_ids)))
                ret += reward

                vehicle_ids=self.env.k.vehicle.get_ids()
                for veh in vehicle_ids:
                    if self.env.k.vehicle.get_type(veh)=="bus":
                        bus_speeds.append(self.env.k.vehicle.get_speed(veh))
                        nonRL_speeds.append(self.env.k.vehicle.get_speed(veh))
                    if self.env.k.vehicle.get_type(veh)=="human":
                        human_speeds.append(self.env.k.vehicle.get_speed(veh))
                        nonRL_speeds.append(self.env.k.vehicle.get_speed(veh))
                    if self.env.k.vehicle.get_type(veh)=="rl":
                        RL_speeds.append(self.env.k.vehicle.get_speed(veh))
                # only include non-empty speeds
                if RL_vel:
                    RL_vel.append(np.mean(RL_speeds))
                    
                bus_vel.append(np.mean(bus_speeds))
                human_vel.append(np.mean(human_speeds))
                nonRL_vel.append(np.mean(nonRL_speeds))

                if Times[i]<j:
                    Times[i]=j

                # Compute the results for the custom callables.
                for (key, lambda_func) in self.custom_callables.items():
                    custom_vals[key].append(lambda_func(self.env))

                if done:
                    break

            # Store the information from the run in info_dict.
            outflow = self.env.k.vehicle.get_outflow_rate(int(500))
            info_dict["returns"].append(ret)
            info_dict["velocities"].append(np.mean(vel))
            info_dict["outflows"].append(outflow)
            for key in custom_vals.keys():
                info_dict[key].append(np.mean(custom_vals[key]))
            

            mean_bus_speed.append(np.mean(bus_vel))
            mean_nonRL_vehicle_speed.append(np.mean(nonRL_vel))
            mean_human_vehicle_speed.append(np.mean(human_vel))
            mean_RL_vehicle_speed.append(np.mean(RL_vel))


            print("Round {0}, return: {1}".format(i, ret))

            AverageReward.append(ret)

            # Save emission data at the end of every rollout. This is skipped
            # by the internal method if no emission path was specified.
            if self.env.simulator == "traci":
                self.env.k.simulation.save_emission(run_id=i)

        # Print the averages/std for all variables in the info_dict.
        for key in info_dict.keys():
            print("Average, std {}: {}, {}".format(
                key, np.mean(info_dict[key]), np.std(info_dict[key])))

        print("\nBus speed, mean (m/s):")
        print(mean_bus_speed)
        print("\nHuman speed, mean (m/s):")
        print(mean_human_vehicle_speed)
        print("\nNonRL speed, mean (m/s):")
        print(mean_nonRL_vehicle_speed)
        print("\nRL speed, mean (m/s):")
        print(mean_RL_vehicle_speed)

        print("Average Reward:")
        print(AverageReward)
        print("Time:")
        print(Times)

        print("Total time:", time.time() - t)
        print("steps/second:", np.mean(times))
        self.env.terminate()


        FinalTimes=[]
        FinalRewards=[]
        FinalRLVelocities=[]
        FinalBusVelocities=[]
        FinalHumanVelocities=[]

        o=0
        for kray in Times:
            if kray>398:
                FinalTimes.append(kray)
                FinalRewards.append(AverageReward[o])
                FinalRLVelocities.append(mean_RL_vehicle_speed[o])
                FinalBusVelocities.append(mean_bus_speed[o])
                FinalHumanVelocities.append(mean_human_vehicle_speed[o])

            o+=1
        print("Final Rewards:")
        for itemu in FinalRewards:
            print(itemu)

        print(FinalTimes)

        print(" ")
        print("Average RL: Velocity For simulations above 400:")
        print(np.mean(FinalRLVelocities))
        print("Standard Deviation for RL vehicles above 400:")
        print(np.std(FinalRLVelocities))

        print(" ")
        print("Average Bus Velocity For simulations above 400:")
        print(np.mean(FinalBusVelocities))
        print("Standard Deviation for Bus vehicles above 400:")
        print(np.std(FinalBusVelocities))

        print(" ")
        print("Average human Velocity For simulations above 400:")
        print(np.mean(FinalHumanVelocities))
        print("Standard Deviation for Human vehicles above 400:")
        print(np.std(FinalHumanVelocities))

        print(" ")
        print("Average Reward For simulations above 400:")
        print(np.mean(FinalRewards))
        print("Standard Deviation for Rewards vehicles above 400:")
        print(np.std(FinalRewards))

        return info_dict
