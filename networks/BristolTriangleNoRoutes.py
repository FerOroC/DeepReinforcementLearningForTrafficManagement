"""Contains the bottleneck network class."""

from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.networks.base import Network
import numpy as np
from numpy import linspace, pi, sin, cos

SCALING = 1


class BristolTriangleNetworkNoRoutes(Network):


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
                 {'id': 'NE_1', 'x': 366.82, 'y': 464.45},
                 {'id': 'TL_NW', 'x': 415.17, 'y': 389.51},
                 {'id': 'TL_M', 'x': 533.53, 'y': 304.73, 'type':'traffic_light'},
                 {'id': 'NE_2', 'x': 635.38, 'y': 276.86, 'type':'traffic_light','shape':"650.37,275.12 649.67,272.00 648.89,270.38 645.02,265.28 642.08,267.77 639.94,269.74 638.14,271.32 636.20,272.66 633.67,273.89 630.08,275.14 631.96,281.26"},
                 {'id': 'NE_3', 'x': 660.61, 'y': 271.17, 'shape':"663.65,271.81 662.60,268.79 661.78,268.90 661.45,268.79 661.19,268.57 660.98,268.25 660.84,267.82 658.89,268.28 658.94,269.18 658.80,269.53 658.57,269.81 658.24,270.02 657.80,270.16 658.50,273.29 660.44,272.84 661.11,272.66 661.77,272.45 662.57,272.18"},
                 {'id': 'NE_4', 'x': 685.81, 'y': 261.26},
                 {'id': 'NE_5', 'x': 706.13, 'y': 252.52},
                 {'id': 'NE_6', 'x': 751.88, 'y': 235.22},

                 {'id': 'EW_1', 'x': 659.13, 'y': 264.97, 'shape':"658.91,268.34 660.85,267.87 660.82,267.04 660.96,266.73 661.20,266.50 661.55,266.33 661.99,266.23 661.61,263.06 656.50,263.79 657.03,266.95 657.90,266.99 658.24,267.17 658.52,267.45 658.75,267.84"},
                                  
                 {'id': 'NS_1', 'x': 648.23, 'y': 266.81, 'type':'traffic_light', 'shape':"659.51,266.53 658.97,263.37 657.71,261.07 652.95,256.78 633.51,268.52 634.22,271.64 634.94,273.18 638.94,278.18"},
                 {'id': 'NS_2', 'x': 651.93, 'y': 262.70},

                 {'id': 'SN_1', 'x': 337.67, 'y': 261.98},
                 {'id': 'SN_2', 'x': 400.47, 'y': 335.93},
                 {'id': 'SN_3', 'x': 402.57, 'y': 355.25},
                 {'id': 'SN_4', 'x': 366.93, 'y': 435.64},
                
                 {'id': 'SW_1', 'x': 707.28, 'y': 192.19},
                 {'id': 'SW_2', 'x': 662.72, 'y': 246.83, 'shape':'670.21,247.78 660.29,239.68 645.28,256.87 649.31,261.84 658.12,249.09 663.92,251.81 665.73,249.91 666.77,249.61 667.88,249.36 669.03,248.85'},
                 {'id': 'SW_3', 'x': 655.06, 'y': 253.05, 'shape':'661.49,251.95 657.46,246.98 655.46,248.03 654.52,248.08 653.61,247.81 652.75,247.23 651.92,246.33 649.38,248.28 650.50,250.46 650.61,251.50 650.41,252.52 649.92,253.51 649.13,254.47 653.67,258.98 655.25,257.40 656.44,256.24 657.44,255.31 658.47,254.42 659.75,253.36'},     
                 {'id': 'SW_4', 'x': 646.30, 'y': 261.33, 'shape':'652.11,263.77 652.58,261.83 652.21,260.45 648.09,255.52 643.08,260.00 647.54,264.59'},
                 {'id': 'SW_5', 'x': 638.34, 'y': 269.06, 'shape':'651.16,267.79 650.45,264.67 649.41,262.77 644.96,258.17 632.84,267.26 634.54,273.43'},

                 {'id': 'SE_1b', 'x': 347.54, 'y': 259.31},  
                 {'id': 'SE_1c', 'x': 346.94, 'y': 240.91},         #Added to make network closed *.              
                 {'id': 'TL_SW', 'x': 405.09, 'y': 291.09},
                 {'id': 'SE_2', 'x': 456.07, 'y': 293.85},
                 {'id': 'SE_3', 'x': 470.75, 'y': 293.76},
                 {'id': 'SE_4', 'x': 491.80, 'y': 293.37},

                 {'id': 'SN_C1', 'x': 571.60, 'y': 384.27},
                 {'id': 'NS_C1', 'x': 521.73, 'y': 269.51},
                 {'id': 'NS_C2', 'x': 514.30, 'y': 241.00},

                 {'id': 'NS_C3', 'x': 465.68, 'y': 254.04},

                 {'id': 'EW_C1', 'x': 622.76, 'y': 211.47}
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
        res = 40

        edges = [
                {'id': 'edge_1', 'from': 'NE_1', 'to': 'TL_NW', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'shape': [(365.81,452.09), (373.27,442.93), (379.76,436.48), (386.63,429.94), (407.02,411.95)], 'spreadType': 'center'},         #top, west to east
                {'id': 'edge_2', 'from': 'TL_NW', 'to': 'TL_M', 'length': 145.59,
                'numLanes': 2, 'type': 'edgeType', 'shape': [(490.38,341.17)], 'spreadType': 'center'},
                {'id': 'edge_12', 'from': 'TL_M', 'to': 'NE_2', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'shape': [(551.30,306.03), (566.13,299.48), (580.32,293.08), (595.85,288.04), (623.31,280.57)], 'spreadType': 'center'},
                {'id': 'edge_13', 'from': 'NE_2', 'to': 'NE_3', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'edge_14', 'from': 'NE_3', 'to': 'NE_4', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'edge_15', 'from': 'NE_4', 'to': 'NE_5', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_29', 'from': 'NE_5', 'to': 'NE_6', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},

                {'id': 'edge_30', 'from': 'NE_6', 'to': 'NE_5', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_26', 'from': 'NE_5', 'to': 'NE_4', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_27', 'from': 'NE_4', 'to': 'EW_1', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'edge_28', 'from': 'EW_1', 'to': 'SW_5', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'center'},                 #{'id': 'edge_28a', 'from': 'NS_1', 'to': 'SW_5', 'length': None, 'numLanes': 1, 'type': 'edgeType', 'spreadType': 'center'},
                # {'id': 'edge_28a', 'from': 'NS_1', 'to': 'SW_5', 'length': None,
                # 'numLanes': 1, 'type': 'edgeType', 'spreadType': 'center'},

                {'id': 'edge_21', 'from': 'NE_2', 'to': 'NS_1', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},                
                {'id': 'edge_22', 'from': 'NS_1', 'to': 'NS_2', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'edge_23', 'from': 'NS_2', 'to': 'SW_2', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center', 'shape':'655.65,258.13 660.33,251.92'},    
                {'id': 'edge_24', 'from': 'SW_2', 'to': 'SW_1', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'right'}, 


                {'id': 'edge_16', 'from': 'SW_1', 'to': 'SW_2', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'right'},                
                 {'id': 'edge_17', 'from': 'SW_2', 'to': 'SW_3', 'length': None,
                 'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center', 'shape':"664.33,248.60 653.69,253.88"},
                {'id': 'edge_18', 'from': 'SW_3', 'to': 'SW_4', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},                
                {'id': 'edge_19', 'from': 'SW_4', 'to': 'SW_5', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'edge_20', 'from': 'SW_5', 'to': 'TL_M', 'length': 110.71,
                'numLanes': 2, 'type': 'edgeType', 'shape':[(627.96,271.92), (614.41,275.64), (584.02,283.81), (565.94,286.83), (555.26,287.77)], 'spreadType': 'center'},

                {'id': 'edge_3', 'from': 'TL_M', 'to': 'SE_4', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center', 'shape':[(521,290.37)]},                
                {'id': 'edge_4', 'from': 'SE_4', 'to': 'SE_3', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'edge_5', 'from': 'SE_3', 'to': 'SE_2', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},                
                {'id': 'edge_6', 'from': 'SE_2', 'to': 'TL_SW', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'edge_7', 'from': 'TL_SW', 'to': 'SE_1b', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'shape':[(391.00,281.65), (382.27,277.98), (372.14,273.29)], 'spreadType': 'center'},
                #*new edge to make network closed

                {'id': 'edge_8', 'from': 'SN_1', 'to': 'TL_SW', 'length': 73.44,
                'numLanes': 1, 'type': 'edgeType', 'shape':[(343.74,265.80), (358.53,274.24), (374.77,283.40), (383.60,288.11)], 'spreadType': 'center'},
                {'id': 'edge_9', 'from': 'TL_SW', 'to': 'SN_2', 'length': 45.08,
                'numLanes': 2, 'type': 'edgeType', 'shape':[(400.93, 317.40)], 'spreadType': 'center'},
                {'id': 'edge_10', 'from': 'SN_2', 'to': 'SN_3', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'edge_11', 'from': 'SN_3', 'to': 'TL_NW', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'edge_25', 'from': 'TL_NW', 'to': 'SN_4', 'length': None,
                'numLanes': 2, 'type': 'edgeType', 'shape': [(402.94,401.23), (398.88,408.25), (388.17,418.34)], 'spreadType': 'center'},
                
                #Capillaries, edge 30
                {'id': 'edge_30', 'from': 'TL_M', 'to': 'SN_C1', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'center'},

                {'id': 'edge_31', 'from': 'NS_C2', 'to': 'NS_C1', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_32', 'from': 'NS_C1', 'to': 'TL_M', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},

                {'id': 'edge_33', 'from': 'TL_M', 'to': 'NS_C1', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_34', 'from': 'NS_C1', 'to': 'NS_C2', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},

                #Second NS

                {'id': 'edge_35', 'from': 'SE_3', 'to': 'NS_C3', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_36', 'from': 'NS_C3', 'to': 'SE_3', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'right'},
                {'id': 'edge_37', 'from': 'SW_3', 'to': 'EW_C1', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'center'},

                #Ammendments to make network closed

                {'id': 'edge_38', 'from': 'EW_C1', 'to': 'NS_C2', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'edge_7b', 'from': 'SE_1b', 'to': 'SE_1c', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'center'},
                {'id': 'edge_7c', 'from': 'SE_1c', 'to': 'NS_C3', 'length': None,
                'numLanes': 1, 'type': 'edgeType', 'spreadType': 'center'},
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

    def specify_connections(self, net_params):
        conn=[]
        num_lanes = 2
        edges_from_b = ['edge_2','edge_11','edge_17','edge_20','edge_21','edge_23']          #Add edges here if they have equal lane mapping
        edges_to_b = ['edge_12','edge_25','edge_18','edge_3','edge_22','edge_24']
        for e_from, e_to in zip(edges_from_b, edges_to_b):
            for i in range(num_lanes):
                x=i
                y=i
                if e_to == 'edge_33':
                    x=1
                    y=0
                conn += [{
                    'from': e_from,
                    'to': e_to,
                    'fromLane': x,
                    'toLane': y
                }]
        edges_from_a = ['edge_2','edge_2','edge_11','edge_14']         #Add edges here if lanes 0 are connected
        edges_to_a = ['edge_3','edge_33','edge_2','edge_15']
        for e_from, e_to in zip(edges_from_a, edges_to_a):
            conn += [{
                'from': e_from,
                'to': e_to,
                'fromLane': 0,
                'toLane': 0
            }]
        edges_from_c = ['edge_2','edge_17']         #Add edges here if lanes 1 and 0 are connected
        edges_to_c = ['edge_30','edge_37']
        for e_from, e_to in zip(edges_from_c, edges_to_c):
            conn += [{
                'from': e_from,
                'to': e_to,
                'fromLane': 1,
                'toLane': 0
            }]
        edges_from_d = ["edge_32"]         #Add edges here if lanes 0 and 1 are connected
        edges_to_d = ["edge_3"]
        for e_from, e_to in zip(edges_from_d, edges_to_d):
            conn += [{
                'from': e_from,
                'to': e_to,
                'fromLane': 0,
                'toLane': 1
            }]
        return conn

    def specify_types(self, net_params):
        """See parent class."""
        types = [{'id': 'edgeType', 'speed': 8.94}]
        return types

    def specify_routes(self, net_params):
        rts = {
            'edge_1': ['edge_1'],
            'edge_2': ['edge_2'],
            'edge_3': ['edge_3'],
            'edge_4': ['edge_4'],
            'edge_5': ['edge_5'],
            'edge_6': ['edge_6'],
            'edge_7': ['edge_7'],
            'edge_8': ['edge_8'],
            'edge_9': ['edge_9'],
            'edge_10': ['edge_10'],
            'edge_11': ['edge_11'],
            'edge_12': ['edge_12'],
            'edge_13': ['edge_13'],
            'edge_14': ['edge_14'],
            'edge_15': ['edge_15'],
            'edge_16': ['edge_16'],
            'edge_17': ['edge_17'],
            'edge_18': ['edge_18'],
            'edge_19': ['edge_19'],
            'edge_20': ['edge_20'],
            'edge_21': ['edge_21'],
            'edge_22': ['edge_22'],
            'edge_23': ['edge_23'],
            'edge_24': ['edge_24'],
            'edge_25': ['edge_25'],
            'edge_26': ['edge_26'],
            'edge_27': ['edge_27'],
            'edge_28': ['edge_28'],                         
            # 'edge_28a': ['edge_28a'],
            'edge_29': ['edge_29'],
            'edge_30': ['edge_30'],
            'edge_31': ['edge_31'],
            'edge_32': ['edge_32'],
            'edge_33': ['edge_33'],
            'edge_34': ['edge_34'],
            'edge_35': ['edge_35'],
            'edge_36': ['edge_36'],
            'edge_37': ['edge_37'],
            #*Ammendments to make network closed
            'edge_38': ['edge_38'],
            'edge_7b': ['edge_7b'],
            'edge_7c': ['edge_7c']
            }
        return rts
