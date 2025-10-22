## what tools do I need
# 1. personal motion calculator between timestamps
# 2. spatial relationship calculation between 2 objects at a given timestamp
# 3. spatial relationship calculation between 2 objects between timestamps

# from langchain_core.tools import tool
# import networkx as nx

# tools = []

# @tools.append
# @tool
# def get_spatial_relationship(first_node_ID: int, second_node_ID: int, nodes: dict) -> tuple:
#     """
#     Get the distance between 2 objects at a given times
#     input:
#         first_node_ID: The object ID of the first object: f"objID_{first_node_ID}"
#         second_node_ID: The object ID of the second object. if not given this is the self node which is at objID_-1: f"objID_{second_node_ID}"
#         nodes: A dictionary with node description of the scene graph at a given time. The keys are the node IDs which are the object IDs. The value is of type GraphNodeNew
#             id: str
#             node_type: str -> This can be text or mask
#             content: If node type is "mask" this will be an image otherwise text
#             caption: Caption for the node
#             bbox: Of the segment in the image given as XYWH format
#             distance2self: the distance to self: ObjID_-1
#             timestamp: str
#     outputs:
#         The spatial distance between the two objects given 
#         d_distance: the distance between object 1 and 2 in depth
#         d_u: the distance between object 1 and 2 in image coordinates u axis
#         d_v: the distance between object 1 and 2 in image coordinates v axis
#     """

#     if second_node_ID is None:
#         second_node_ID = -1

#     rel_distance_to_obj1_from_self = nodes[f'objID_{first_node_ID}'].distance2self
#     rel_distance_to_obj2_from_self = nodes[f'objID_{second_node_ID}'].distance2self

#     d_distance = rel_distance_to_obj1_from_self - rel_distance_to_obj2_from_self

#     [u_1, v_1, width_1, height_1] =  nodes[f'objID_{first_node_ID}'].bbox
#     [u_2, v_2, width_2, height_2] =  nodes[f'objID_{second_node_ID}'].bbox
#     d_u = u_1-u_2
#     d_v = v_1 - v_2

#     return d_distance, d_u, d_v

# @tools.append
# @tool
# def prune_graph(nodes: dict,query_nodes:list) -> dict:
#     """A tool to prune the existing nodes dictionary which can be be quite big
#     inputs:
#         nodes: A dictionary with node description of the scene graph at a given time. The keys are the node IDs which are the object IDs. The value is of type GraphNodeNew
#             id: str
#             node_type: str -> This can be text or mask
#             content: If node type is "mask" this will be an image otherwise text
#             caption: Caption for the node
#             bbox: Of the segment in the image given as XYWH format
#             distance2self: the distance to self: ObjID_-1
#             timestamp: str
#         query_nodes" A list of nodes of interest
#     outputs:
#         relevant query nodes
#     """
#     relevant_nodes = {}
#     for q in query_nodes:
#         if q in nodes.keys():
#             relevant_nodes[q] = nodes[q]
#     return relevant_nodes


# if rel_distance_to_obj1_from_self>rel_distance_to_obj2_from_self:
#     output += f"object {first_node_ID} is behind object {second_node_ID}."
# elif rel_distance_to_obj1_from_self<rel_distance_to_obj2_from_self:
#     output += f"object {first_node_ID} is in front of object {second_node_ID}."
# elif rel_distance_to_obj1_from_self == rel_distance_to_obj2_from_self:
#     output += f"object {first_node_ID} and object {second_node_ID} are of equal distance to me."

from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel

model = LiteLLMModel(
  model_id='ollama_chat/qwen3'
)
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?", reset=True)

agent.python_executor.send_variables({'my_fav_num':5})
agent.run('My favourite number is stored in the variable my_fav_num. what is my favourite number multipled by 5?', reset=False)

import pickle
file_path = '/scratch3/kat049/segment-anything-2-real-time/my_graph.pickle'
with open(file_path, 'rb') as file:
    G = pickle.load(file)