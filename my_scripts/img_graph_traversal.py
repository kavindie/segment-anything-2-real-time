import autogen
from autogen import ConversableAgent
import networkx as nx
import requests
import json

class ImageGraphAgent:
    def __init__(self):
        self.graph = nx.Graph()
        self.descriptions = {}
        
        # Test Ollama connection first
        self.test_ollama_connection()
        
        # Configure the agent with correct Ollama settings
        config_list = [
            {
                "model": "llama3.1:8b",
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",  # Can be anything for Ollama
                "api_type": "openai"  # This is important
            }
        ]
        
        self.agent = ConversableAgent(
            name="image_graph_agent",
            system_message="""You are an agent that answers questions about images connected by distances. 
            You have access to a graph where nodes are images and edges represent distances.
            Use the available functions to search, navigate, and analyze the graph.""",
            llm_config={
                "config_list": config_list,
                "temperature": 0.1,
                "timeout": 60,
            },
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        # Register functions
        self.register_functions()
    
    def test_ollama_connection(self):
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models['models']]
                print(f"✅ Ollama is running. Available models: {available_models}")
                if 'llama3.1:8b' not in available_models:
                    print("⚠️  llama3.1:8b not found. Pulling it...")
                    import subprocess
                    subprocess.run(["ollama", "pull", "llama3.1:8b"])
            else:
                print("❌ Ollama is not responding correctly")
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama. Make sure 'ollama serve' is running")
            return False
        except Exception as e:
            print(f"❌ Error testing Ollama: {e}")
            return False
        return True
    
    def register_functions(self):
        @self.agent.register_for_execution()
        @self.agent.register_for_llm(description="Search for image nodes matching a description")
        def search_nodes(query: str) -> str:
            """Search for nodes containing the query text"""
            results = []
            for node_id, desc in self.descriptions.items():
                if query.lower() in desc.lower():
                    results.append(f"Node {node_id}: {desc}")
            return "\n".join(results[:5]) if results else "No matching nodes found"
        
        @self.agent.register_for_execution()
        @self.agent.register_for_llm(description="Find shortest path between two nodes")
        def shortest_path(start: str, end: str) -> str:
            """Find the shortest path between two nodes"""
            try:
                if start not in self.graph.nodes or end not in self.graph.nodes:
                    return f"One or both nodes ({start}, {end}) not found in graph"
                    
                path = nx.shortest_path(self.graph, start, end, weight='weight')
                distance = nx.shortest_path_length(self.graph, start, end, weight='weight')
                return f"Shortest path from {start} to {end}: {' → '.join(path)}\nTotal distance: {distance:.1f} units"
            except nx.NetworkXNoPath:
                return f"No path exists between {start} and {end}"
            except Exception as e:
                return f"Error finding path: {str(e)}"
        
        @self.agent.register_for_execution()
        @self.agent.register_for_llm(description="Find nodes within specified distance of a center node")
        def find_nearby(center: str, max_distance: float) -> str:
            """Find all nodes within max_distance of the center node"""
            try:
                if center not in self.graph.nodes:
                    return f"Center node {center} not found in graph"
                    
                distances = nx.single_source_dijkstra_path_length(
                    self.graph, center, cutoff=max_distance, weight='weight'
                )
                
                nearby = []
                for node, dist in distances.items():
                    if node != center and dist <= max_distance:
                        desc = self.descriptions.get(node, "No description")
                        nearby.append(f"{node} ({desc}): {dist:.1f} units away")
                
                return "\n".join(nearby) if nearby else f"No nodes found within {max_distance} units of {center}"
            except Exception as e:
                return f"Error finding nearby nodes: {str(e)}"
    
    def add_image_node(self, node_id: str, description: str):
        """Add an image node to the graph"""
        self.graph.add_node(node_id)
        self.descriptions[node_id] = description
        print(f"Added node {node_id}: {description}")
    
    def add_distance_edge(self, node1: str, node2: str, distance: float):
        """Add a distance edge between two nodes"""
        if node1 in self.graph.nodes and node2 in self.graph.nodes:
            self.graph.add_edge(node1, node2, weight=distance)
            print(f"Added edge {node1} ↔ {node2}: {distance} units")
        else:
            print(f"Cannot add edge: one or both nodes ({node1}, {node2}) don't exist")
    
    def answer_question(self, question: str):
        """Answer a question about the image graph"""
        try:
            print(f"\n🤔 Question: {question}")
            
            # Create a user proxy to handle the conversation
            user_proxy = autogen.UserProxyAgent(
                name="user",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False
            )
            
            # Start the conversation
            response = user_proxy.initiate_chat(
                self.agent,
                message=question,
                max_turns=2
            )
            
            return "Conversation completed - check output above"
            
        except Exception as e:
            return f"Error: {str(e)}"

# Usage example
if __name__ == "__main__":
    # Create the agent
    agent = ImageGraphAgent()
    
    # Add some sample data
    agent.add_image_node("img1", "red car in parking lot")
    agent.add_image_node("img2", "blue hospital building")  
    agent.add_image_node("img3", "green park with trees")
    agent.add_image_node("img4", "yellow school building")
    
    # Add distance edges
    agent.add_distance_edge("img1", "img2", 500.0)  # 500 units
    agent.add_distance_edge("img2", "img3", 300.0)  # 300 units
    agent.add_distance_edge("img1", "img4", 200.0)  # 200 units
    agent.add_distance_edge("img3", "img4", 400.0)  # 400 units
    
    # Test questions
    questions = [
        "What's the shortest path from the red car to the park?",
        "Find all nodes within 600 units of the hospital",
        "Search for nodes containing the word 'building'"
    ]
    
    for question in questions:
        print("\n" + "="*50)
        agent.answer_question(question)