from langgraph.graph import StateGraph, START, END
from langchain_community.tools import BaseTool
from langchain.agents import tool
import networkx as nx
import clip
import torch
from sentence_transformers import SentenceTransformer
import json
import math
from typing import List, Dict
import random
import pickle



from langchain_ollama import ChatOllama
from rosa import ROSA

import sys
sys.path.extend([
    '/scratch3/kat049/segment-anything-2-real-time',
])
from my_scripts.system_prompt import get_prompt
import pickle



def get_llm(streaming: bool = False):
  
    ollama_model = 'qwen2.5:14b'
    ollama_base_url = 'http://localhost:11434/'
 
    ollama_llm = ChatOllama(
        model=ollama_model,
        base_url=ollama_base_url,
        temperature=0,
        keep_alive="5m",
        streaming=streaming,
        num_ctx=8192,   # Increase context length for better tool understanding
    )
    return ollama_llm

class ImageGraphAgent():
    def __init__(self):
        tool_dict = dict(
            closest_image_node= self._create_closest_image_node(),
            sample_random_nodes = self._create_sample_random_nodes()
        )
        self.tools = list(tool_dict.tools.values())

        self.agent = ROSA(
            ros_version=2,
            llm=get_llm(),
            prompts=get_prompt(),
            tools=self.tools
        )

        with open('/scratch3/kat049/segment-anything-2-real-time/my_graph.pickle', 'rb') as file:
            self.G = pickle.load(file)
        nodes_org = self.G.nodes
        self.nodes = [{"node_id": key, "node_description": data['caption']} 
                for key, data in nodes_org.items()]

    def _create_closest_image_node(self):
        @tool
        def closest_image_node(query: str, candidate_nodes: List[Dict[str, str]]=self.nodes) -> str:
            """Search for image nodes that match a text description from candidate nodes which are already provided
            
            Args:
                query (str): Text description to search for in image nodes
                candidate_nodes (List[Dict[str, str]]): List of candidate nodes, each with format:
                                                        [{"node_id": "node_123", "description": "image description"}, ...]
                                                        (default: provided)
                
            Returns:
                str: JSON string containing the most similar image node with format:
                        [{"node_id": "node_123", "description": "image description"}, ...]
                        Returns the top match from the candidate nodes
            """

            text_model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = text_model.encode([query.lower()])
            best_sim = -math.inf
            result = {"node_id": None, "description": None}
            
            for node in self.nodes:
                    node_id = node["node_id"]
                    desc = node["description"]
                    desc_embedding = text_model.encode([desc.lower()])
                    sim = desc_embedding@query_embedding
                    if sim > best_sim:
                        result = {"node_id": node_id, "description": desc}
            
            return json.dumps(result) 
        return closest_image_node

    def _create_sample_random_nodes(self):
        @tool
        def sample_random_nodes(all_nodes: Dict[str, str]=nodes, sample_size: int = 10) -> str:
            """Randomly sample nodes from all available image nodes
            
            Args:
                all_nodes (Dict[str, str]): Dictionary mapping node_id to description
                                            {"node_123": "image description", ...}
                sample_size (int): Number of nodes to sample (default: 10)
                
            Returns:
                str: JSON string containing list of randomly sampled nodes with format:
                        [{"node_id": "node_123", "description": "image description"}, ...]
            """
            node_items = list(all_nodes.items())
            sampled = random.sample(node_items, min(sample_size, len(node_items)))
            
            results = [{"node_id": node_id, "description": desc} for node_id, desc in sampled]
            
            return json.dumps(results)
        return sample_random_nodes
# class ImageGraphProcessor:
#     def __init__(self):
#         self.graph = nx.Graph()
#         self.image_embeddings = {}
#         self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.image_descriptions = {}
    
#     def add_image_node(self, node_id: str, image_path: str, description: str = ""):
#         self.graph.add_node(node_id, image_path=image_path)
#         if description:
#             self.image_descriptions[node_id] = description
    
#     def add_distance_edge(self, node1: str, node2: str, distance: float):
#         self.graph.add_edge(node1, node2, weight=distance)

# # Define tools as functions (easier approach)
# processor = ImageGraphProcessor()

# class SearchImageNodesTool(BaseTool):
#     name = "search_image_nodes"
#     description = "Search for image nodes that match a text description"
    
#     def _run(self, query: str) -> str:
#         query_embedding = processor.text_model.encode([query])
#         results = []
        
#         for node_id, desc in processor.image_descriptions.items():
#             if query.lower() in desc.lower():
#                 results.append({"node_id": node_id, "description": desc})
        
#         return json.dumps(results[:5])  # Return top 5 matches

# class FindShortestPathTool(BaseTool):
#     name = "find_shortest_path"
#     description = "Find shortest path between two nodes"
    
#     def _run(self, start_node: str, end_node: str) -> str:
#         try:
#             path = nx.shortest_path(processor.graph, start_node, end_node, weight='weight')
#             total_distance = nx.shortest_path_length(processor.graph, start_node, end_node, weight='weight')
#             return json.dumps({"path": path, "total_distance": total_distance})
#         except nx.NetworkXNoPath:
#             return json.dumps({"error": "No path found between nodes"})

# class FindNearbyNodesTool(BaseTool):
#     name = "find_nearby_nodes"
#     description = "Find all nodes within a specified distance from a center node"
    
#     def _run(self, center_node: str, max_distance: float) -> str:
#         try:
#             distances = nx.single_source_dijkstra_path_length(
#                 processor.graph, center_node, cutoff=max_distance, weight='weight'
#             )
#             nearby = {node: dist for node, dist in distances.items() if dist <= max_distance}
#             return json.dumps(nearby)
#         except nx.NetworkXError:
#             return json.dumps({"error": f"Node {center_node} not found in graph"})

# # Initialize tools
# tools = [
#     SearchImageNodesTool(),
#     FindShortestPathTool(), 
#     FindNearbyNodesTool()
# ]




print(agent.invoke(f"is there a horse?"))