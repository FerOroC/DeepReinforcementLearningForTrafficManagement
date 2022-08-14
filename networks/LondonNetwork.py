"""Contains the bottleneck network class."""

from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.networks.base import Network
import numpy as np
from numpy import linspace, pi, sin, cos

SCALING = 1


class LondonClosed(Network):


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
                 {'id': 'S_1', 'x': 427.41, 'y': 152.54},
                 {'id': 'S_2', 'x': 474.97, 'y': 167.35},
                 {'id': 'S_3', 'x': 529.01, 'y': 184.61},
                 {'id': 'S_4', 'x': 568.76, 'y': 198.26},
                 {'id': 'S_5', 'x': 599.64, 'y': 208.86},
                 {'id': 'S_6', 'x': 672.62, 'y': 233.28},
                 {'id': 'S_7', 'x': 728.35, 'y': 248.69},

                 {'id': 'S_8', 'x': 392.56, 'y': 255.93},
                 {'id': 'S_9', 'x': 441.89, 'y': 270.94},
                 {'id': 'S_10', 'x': 494.53, 'y': 286.99},
                 {'id': 'S_11', 'x': 567.58, 'y': 311.32},
                 {'id': 'S_12', 'x': 638.80, 'y': 334.57},
                 {'id': 'S_13', 'x': 691.32, 'y': 351.68},

                 {'id': 'S_14', 'x': 351.16, 'y': 377.83},
                 {'id': 'S_15', 'x': 453.14, 'y': 409.62},
                 {'id': 'S_16', 'x': 482.10, 'y': 418.32},
                 {'id': 'S_17', 'x': 599.19, 'y': 454.76},
                 {'id': 'S_18', 'x': 649.29, 'y': 470.00},          

                 {'id': 'S_18b', 'x': 326.46, 'y': 453.53},  
                 {'id': 'S_19', 'x': 428.96, 'y': 485.11},
                 {'id': 'S_20', 'x': 479.64, 'y': 500.82},
                 {'id': 'S_21', 'x': 573.69, 'y': 530.52},
                 {'id': 'S_22', 'x': 624.15, 'y': 545.25},
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
                {'id': 'edge_WE_1', 'from': 'S_1', 'to': 'S_2', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_WE_2', 'from': 'S_2', 'to': 'S_3', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_WE_3', 'from': 'S_3', 'to': 'S_4', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_WE_4', 'from': 'S_4', 'to': 'S_5', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_WE_5', 'from': 'S_5', 'to': 'S_6', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_WE_6', 'from': 'S_6', 'to': 'S_7', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},

                {'id': 'edge_EW_1', 'from': 'S_2', 'to': 'S_1', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_2', 'from': 'S_3', 'to': 'S_2', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_3', 'from': 'S_4', 'to': 'S_3', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_4', 'from': 'S_5', 'to': 'S_4', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_5', 'from': 'S_6', 'to': 'S_5', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_6', 'from': 'S_7', 'to': 'S_6', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},


                {'id': 'edge_WE_8', 'from': 'S_8', 'to': 'S_9', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_WE_9', 'from': 'S_9', 'to': 'S_10', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_WE_10', 'from': 'S_10', 'to': 'S_11', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_WE_11', 'from': 'S_11', 'to': 'S_12', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_WE_12', 'from': 'S_12', 'to': 'S_13', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                
                {'id': 'edge_EW_8', 'from': 'S_9', 'to': 'S_8', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_9', 'from': 'S_10', 'to': 'S_9', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_10', 'from': 'S_11', 'to': 'S_10', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_11', 'from': 'S_12', 'to': 'S_11', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_12', 'from': 'S_13', 'to': 'S_12', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},


                {'id': 'edge_WE_13', 'from': 'S_14', 'to': 'S_15', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_WE_14', 'from': 'S_15', 'to': 'S_16', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'edge_WE_15', 'from': 'S_16', 'to': 'S_17', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'edge_WE_16', 'from': 'S_17', 'to': 'S_18', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},

                {'id': 'edge_EW_13', 'from': 'S_15', 'to': 'S_14', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_14', 'from': 'S_16', 'to': 'S_15', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_15', 'from': 'S_17', 'to': 'S_16', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_16', 'from': 'S_18', 'to': 'S_17', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},


                {'id': 'edge_WE_17', 'from': 'S_18b', 'to': 'S_19', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_WE_18', 'from': 'S_19', 'to': 'S_20', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_WE_19', 'from': 'S_20', 'to': 'S_21', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_WE_20', 'from': 'S_21', 'to': 'S_22', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},

                {'id': 'edge_EW_17', 'from': 'S_19', 'to': 'S_18b', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_18', 'from': 'S_20', 'to': 'S_19', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_19', 'from': 'S_21', 'to': 'S_20', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_EW_20', 'from': 'S_22', 'to': 'S_21', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},


                {'id': 'edge_NS_1', 'from': 'S_18b', 'to': 'S_14', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_NS_2', 'from': 'S_14', 'to': 'S_8', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_NS_3', 'from': 'S_8', 'to': 'S_1', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},

                {'id': 'edge_SN_1', 'from': 'S_14', 'to': 'S_18b', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_SN_2', 'from': 'S_8', 'to': 'S_14', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_SN_3', 'from': 'S_1', 'to': 'S_8', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},


                {'id': 'edge_SN_4', 'from': 'S_9', 'to': 'S_2', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},

                {'id': 'edge_NS_4', 'from': 'S_2', 'to': 'S_9', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},


                {'id': 'edge_NS_5', 'from': 'S_19', 'to': 'S_15', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_NS_6', 'from': 'S_15', 'to': 'S_10', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_NS_7', 'from': 'S_10', 'to': 'S_3', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},

                {'id': 'edge_SN_5', 'from': 'S_15', 'to': 'S_19', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_SN_6', 'from': 'S_10', 'to': 'S_15', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_SN_7', 'from': 'S_3', 'to': 'S_10', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'right'},


                {'id': 'edge_NS_8', 'from': 'S_21', 'to': 'S_17', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_NS_9', 'from': 'S_17', 'to': 'S_12', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_NS_10', 'from': 'S_12', 'to': 'S_6', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'right'},

                {'id': 'edge_SN_8', 'from': 'S_17', 'to': 'S_21', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_SN_9', 'from': 'S_12', 'to': 'S_17', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_SN_10', 'from': 'S_6', 'to': 'S_12', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},


                {'id': 'edge_NS_11', 'from': 'S_22', 'to': 'S_18', 'length': None,
                'numLanes': 1, 'type': 'destination', 'spreadType': 'right'},
                {'id': 'edge_NS_12', 'from': 'S_18', 'to': 'S_13', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_NS_13', 'from': 'S_13', 'to': 'S_7', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},

                {'id': 'edge_SN_11', 'from': 'S_18', 'to': 'S_22', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_SN_12', 'from': 'S_13', 'to': 'S_18', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_SN_13', 'from': 'S_7', 'to': 'S_13', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},

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
        types = [
            {'id': 'edgeType', 'speed': 8},
            {'id': 'destination', 'speed': 8}
        ]
        return types

    def specify_routes(self, net_params):
        rts = {
            'edge_WE_1': ['edge_WE_1'],
            'edge_WE_2': ['edge_WE_2'],
            'edge_WE_3': ['edge_WE_3'],
            'edge_WE_4': ['edge_WE_4'],
            'edge_WE_5': ['edge_WE_5'],
            'edge_WE_6': ['edge_WE_6'],
            'edge_WE_8': ['edge_WE_8'],
            'edge_WE_9': ['edge_WE_9'],
            'edge_WE_10': ['edge_WE_10'],
            'edge_WE_11': ['edge_WE_11'],
            'edge_WE_12': ['edge_WE_12'],
            'edge_WE_13': ['edge_WE_13'],
            'edge_WE_14': ['edge_WE_14'],
            'edge_WE_15': ['edge_WE_15'],
            'edge_WE_16': ['edge_WE_16'],
            'edge_WE_17': ['edge_WE_17'],
            'edge_WE_18': ['edge_WE_18'],
            'edge_WE_19': ['edge_WE_19'],
            'edge_WE_20': ['edge_WE_20'],
            'edge_EW_1': ['edge_EW_1'],
            'edge_EW_2': ['edge_EW_2'],
            'edge_EW_3': ['edge_EW_3'],
            'edge_EW_4': ['edge_EW_4'],
            'edge_EW_5': ['edge_EW_5'],
            'edge_EW_6': ['edge_EW_6'],
            'edge_EW_8': ['edge_EW_8'],
            'edge_EW_9': ['edge_EW_9'],
            'edge_EW_10': ['edge_EW_10'],
            'edge_EW_11': ['edge_EW_11'],
            'edge_EW_12': ['edge_EW_12'],
            'edge_EW_13': ['edge_EW_13'],
            'edge_EW_14': ['edge_EW_14'],
            'edge_EW_15': ['edge_EW_15'],
            'edge_EW_16': ['edge_EW_16'],
            'edge_EW_17': ['edge_EW_17'],
            'edge_EW_18': ['edge_EW_18'],
            'edge_EW_19': ['edge_EW_19'],
            'edge_EW_20': ['edge_EW_20'],
            'edge_NS_1': ['edge_NS_1'],
            'edge_NS_2': ['edge_NS_2'],
            'edge_NS_3': ['edge_NS_3'],
            'edge_NS_4': ['edge_NS_4'],
            'edge_NS_5': ['edge_NS_5'],
            'edge_NS_6': ['edge_NS_6'],
            'edge_NS_7': ['edge_NS_7'],
            'edge_NS_8': ['edge_NS_8'],
            'edge_NS_9': ['edge_NS_9'],
            'edge_NS_10': ['edge_NS_10'],
            'edge_NS_11': ['edge_NS_11'],
            'edge_NS_12': ['edge_NS_12'],
            'edge_NS_13': ['edge_NS_13'],
            'edge_SN_1': ['edge_SN_1'],
            'edge_SN_2': ['edge_SN_2'],
            'edge_SN_3': ['edge_SN_3'],
            'edge_SN_4': ['edge_SN_4'],
            'edge_SN_5': ['edge_SN_5'],
            'edge_SN_6': ['edge_SN_6'],
            'edge_SN_7': ['edge_SN_7'],
            'edge_SN_8': ['edge_SN_8'],
            'edge_SN_9': ['edge_SN_9'],
            'edge_SN_10': ['edge_SN_10'],
            'edge_SN_11': ['edge_SN_11'],
            'edge_SN_12': ['edge_SN_12'],
            'edge_SN_13': ['edge_SN_13'],
        }
        return rts

    def specify_connections(self, net_params):
        conn=[]
        edges_from_a = ['edge_SN_6','edge_SN_6','edge_SN_6','edge_NS_10','edge_NS_10','edge_SN_7','edge_SN_7','edge_SN_7','edge_WE_15','edge_WE_15','edge_WE_15','edge_NS_9','edge_NS_9','edge_NS_9','edge_NS_8','edge_NS_8','edge_NS_8','edge_SN_5','edge_SN_5']         #Add edges here if lanes 0 are connected
        edges_to_a = ['edge_EW_13','edge_WE_14','edge_SN_5','edge_EW_5','edge_WE_6','edge_EW_9','edge_SN_6','edge_WE_10','edge_SN_8','edge_WE_16','edge_NS_9','edge_EW_11','edge_NS_10','edge_WE_12','edge_EW_15','edge_NS_9','edge_WE_16','edge_WE_18','edge_EW_17']
        for e_from, e_to in zip(edges_from_a, edges_to_a):
            conn += [{
                'from': e_from,
                'to': e_to,
                'fromLane': 0,
                'toLane': 0
            }]
        edges_from_c = ['edge_SN_6','edge_SN_6','edge_SN_6','edge_NS_10','edge_NS_10','edge_SN_7','edge_SN_7','edge_SN_7','edge_WE_15','edge_WE_15','edge_WE_15','edge_NS_9','edge_NS_9','edge_NS_9','edge_NS_8','edge_NS_8','edge_NS_8','edge_SN_5','edge_SN_5']         #Add edges here if lanes 1 and 0 are connected
        edges_to_c = ['edge_EW_13','edge_WE_14','edge_SN_5','edge_EW_5','edge_WE_6','edge_EW_9','edge_SN_6','edge_WE_10','edge_SN_8','edge_WE_16','edge_NS_9','edge_EW_11','edge_NS_10','edge_WE_12','edge_EW_15','edge_NS_9','edge_WE_16','edge_WE_18','edge_EW_17']
        for e_from, e_to in zip(edges_from_c, edges_to_c):
            conn += [{
                'from': e_from,
                'to': e_to,
                'fromLane': 1,
                'toLane': 0
            }]
        return conn

    def additional_command(self):
        """This command will be used to reroute vehicles to their startting positions
        if they have reached their final destination"""
        print(self.k.vehicle.get_edge(veh_id))
        for veh_id in self.k.vehicle.get_rl_ids():
            if self.k.vehicle.get_edge(veh_id)=="edge_NS_11":
                self._reroute_at_destination(veh_id)
    
    def _reroute_at_destination(self,veh_id):
        """Reroute RL vehicle id back to start position if destination reached."""

        type_id=self.k.vehicle.get_type(veh_id)

        self.k.vehicle.remove(veh_id)

        self.k.vehicle.add(
            veh_id=veh_id,
            edge="edge_WE_1",
            type_id=str(type_id),
            lane="0",
            pos="0",
            speed="max"
        )