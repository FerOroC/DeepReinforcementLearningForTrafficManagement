import random
import numpy as np
from gym.spaces.box import Box 

from flow.controllers.base_routing_controller import BaseRouter

class RLRoutingController(BaseRouter):
    def __init__(self, veh_id, router_params):
        """Instantiate an RL Controller."""
        BaseRouter.__init__(
            self,
            veh_id,
            router_params
            )
    
    def choose_route(self, env):
        """Pass, as this is never called; required to override abstractmethod."""
        pass