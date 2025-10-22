# import networkx as nx
# from smolagents import CodeAgent, HfApiModel
# import pickle
# import json
# import random

# # Initialize Ollama model
# model = HfApiModel(model_id="ollama/llama3.2:3b")

# # Create agent with NetworkX tools
# agent = CodeAgent(
#     tools=[],
#     model=model,
#     additional_authorized_imports=["networkx", "pickle", "json", "random"]
# )

# # Load your graph
# with open('/scratch3/kat049/segment-anything-2-real-time/my_graph.pickle', 'rb') as file:
#     G = pickle.load(file)

# # Make graph accessible globally for the agent
# globals()['G'] = G

# # Example usage
# query = """
# I have a NetworkX graph G loaded with image nodes. Each node has 'content' attribute with descriptions.
# Can you help me find nodes that contain 'horse' in their content?

# Use this code structure:
# ```python
# import networkx as nx
# results = []
# for node_id, data in G.nodes(data=True):
#     if 'horse' in data.get('content', '').lower():
#         results.append({'node_id': node_id, 'content': data['content']})
# print(f"Found {len(results)} nodes with horses")
# for result in results[:3]:  # Show first 3
#     print(f"Node {result['node_id']}: {result['content'][:100]}...")
# ```
# """

# # Run the agent
# response = agent.run("Is there a horse?")
# print(response)
import sys
sys.path.extend([
    '/scratch3/kat049/segment-anything-2-real-time',
])
from smolagents.agents import CodeAgent
from smolagents.models import TransformersModel
from smolagents.tools import Tool
from my_scripts.img_graph_tools import get_3Dfromuv

model = TransformersModel(model_id="llama3")
tools = [Tool(name="get_3Dfromuv", func=get_3Dfromuv, description="Back-projects 2D (u,v,depth) to 3D")]

agent = CodeAgent(tools=tools, model=model)
agent.run("Get 3D coordinates for u=1000, v=800, depth=2.5")
