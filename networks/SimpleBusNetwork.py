"""Contains the bottleneck network class."""

from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.networks.base import Network
import numpy as np

SCALING=1
class BusLaneNetwork(Network):
    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Instantiate the network class."""
        self.nodes_dict = dict()

        super().__init__(name, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        """See parent class."""
        length = 0
        edgestarts = []
        for edge in self.edges:
            # the current edge starts where the last edge ended
            edgestarts.append((edge['id'], length))
            # increment the total length of the network with the length of the
            # current edge
            length += float(edge['length'])

        return edgestarts

    def specify_nodes(self, net_params):
        """See parent class."""
        nodes = [
                 {'id': 'N_1', 'x': 0.00, 'y': 150.00},
                 {'id': 'N_2', 'x': 150.00, 'y': 150.00},
                 {'id': 'S_1', 'x': 0.00, 'y': 0.00},
                 {'id': 'S_2', 'x': 150.00, 'y': 0.00},
        ]
        for node in nodes:
            self.nodes_dict[node['id']] = np.array([node['x'] * SCALING,
                                                    node['y'] * SCALING])

        for node in nodes:
            node['x'] = node['x'] * SCALING
            node['y'] = node['y'] * SCALING

        return nodes
    
    def specify_edges(self, net_params):
        """See parent class."""

        edges = [
                {'id': 'gneE1', 'from': 'N_2', 'to': 'N_1', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'gneE2', 'from': 'N_1', 'to': 'S_1', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'gneE3', 'from': 'S_1', 'to': 'S_2', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'gneE4', 'from': 'S_2', 'to': 'N_2', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},
        ]
        for edge in edges:
            if ('shape' in edge) and (type(edge['length']) is None):
                edge['length']=edge['length']
                edge['shape'] = [(x * SCALING, y * SCALING)
                                 for x, y in edge['shape']]
                print("none")
            elif 'shape' in edge and (type(edge['length']) is int):
                edge['length'] = sum(
                    [np.sqrt((edge['shape'][i][0] - edge['shape'][i+1][0])**2 +
                             (edge['shape'][i][1] - edge['shape'][i+1][1])**2)
                     * SCALING for i in range(len(edge['shape'])-1)])
                edge['shape'] = [(x * SCALING, y * SCALING)
                                 for x, y in edge['shape']]
            else:
                edge['length'] = np.linalg.norm(
                    self.nodes_dict[edge['to']] -
                    self.nodes_dict[edge['from']])
        return edges

    def specify_types(self, net_params):
        """See parent class."""
        types = [{'id': 'edgeType', 'speed': 8}]
        return types

    def specify_routes(self, net_params):
        rts = {
            'gneE1':['gneE1'],
            'gneE2':['gneE2'],
            'gneE3':['gneE3'],
            'gneE4':['gneE4']
        }
        return rts