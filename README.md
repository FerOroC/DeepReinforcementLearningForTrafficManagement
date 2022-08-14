# Deep Reinforcement Learning for Connected and Autonomous Vehicle traffic management controller

Using the flow framework developed by UC Berkeley a centralised agent, which acts as the Connected and Autonomous Vehicles' (CAVs) lane-changing and acceleration controller, is trained to maximise the average velocity of vehicles whilst minimising delays to high-priority vehicles in the network. 

I have uploaded this here to help anyone looking to use the flow framework to undertake Deep Reinforcement Learning (DRL) research. Attached are the simplified versions of the network, environment, and training files used for my thesis. Thank you to the team at UC Berkeley for your contributions in the field of RL, as well as your ongoing assistance in the flow slack channel. Follow the reference underneath for more information on their work.

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol. abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465

# Getting Started

Windows not supported by certain packages; only Linux. 

For local installatoins, follow the instructions in: https://flow.readthedocs.io/en/latest/flow_setup.html#local-installation-of-flow, changing package versions where needed to meet dependencies. 

# Training an Agent

To train a particular agent, move to the flow directory and enter the following

```
python examples/train.py agentandenvironmentfile
```

Within the agentandenvironmentfile, we would reference the road network file, an environment class, a configuration file, and the simulation parameters. Can be either a single agent, or a multi agent approach. The naming convention for the agentandenvironmentfile is: 

```
singleagent_(OR multiagent_)environment
```

The train.py file will automatically detect the singleagent or multiagent training approach depending on the name of the file. An example can be found in the file **singleagent_BusLaneController.py**. Regarding the urban road networks, their designs can be found in the **BusLaneNetwork.py**, **LondonGridNetwork.py**, and **BristolTriangle.py**. All controller files can be found in the flow framework repository.
