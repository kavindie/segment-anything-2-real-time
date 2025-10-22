from autogen_ext.models.ollama import OllamaChatCompletionClient
import networkx as nx
import asyncio
import re
from typing import Dict, List
from autogen_core.models import LLMMessage, SystemMessage,  UserMessage, AssistantMessage, FunctionExecutionResultMessage

class ImageGraphProcessor:
    def __init__(self):
        self.graph = nx.Graph()
        self.descriptions = {}
        self.image_paths = {}
    
    def add_image_node(self, node_id: str, description: str, image_path: str = None):
        self.graph.add_node(node_id)
        self.descriptions[node_id] = description
        if image_path:
            self.image_paths[node_id] = image_path
        print(f"✅ Added node {node_id}: {description}")
    
    def add_distance_edge(self, node1: str, node2: str, distance: float):
        if node1 in self.graph.nodes and node2 in self.graph.nodes:
            self.graph.add_edge(node1, node2, weight=distance)
            print(f"🔗 Added edge {node1} ↔ {node2}: {distance} units")
        else:
            print(f"⚠️ Cannot add edge: one or both nodes ({node1}, {node2}) don't exist")

# Global processor instance
processor = ImageGraphProcessor()

# Tool functions
def search_image_nodes(query: str) -> str:
    results = []
    for node_id, desc in processor.descriptions.items():
        if query.lower() in desc.lower():
            results.append(f"Node {node_id}: {desc}")
    return "\n".join(results[:5]) if results else "No matching nodes found"

def find_shortest_path(start_node: str, end_node: str) -> str:
    """Find shortest path between two nodes"""
    try:
        if start_node not in processor.graph.nodes or end_node not in processor.graph.nodes:
            return f"One or both nodes ({start_node}, {end_node}) not found in graph"
            
        path = nx.shortest_path(processor.graph, start_node, end_node, weight='weight')
        distance = nx.shortest_path_length(processor.graph, start_node, end_node, weight='weight')
        
        # Add descriptions to path
        path_with_desc = []
        for node in path:
            desc = processor.descriptions.get(node, "No description")
            path_with_desc.append(f"{node}({desc})")
        
        return f"Shortest path: {' → '.join(path_with_desc)}\nTotal distance: {distance:.1f} units"
    except nx.NetworkXNoPath:
        return f"No path exists between {start_node} and {end_node}"
    except Exception as e:
        return f"Error finding path: {str(e)}"

def find_nearby_nodes(center_node: str, max_distance: float) -> str:
    """Find all nodes within specified distance of center node"""
    try:
        if center_node not in processor.graph.nodes:
            return f"Center node {center_node} not found in graph"
            
        distances = nx.single_source_dijkstra_path_length(
            processor.graph, center_node, cutoff=max_distance, weight='weight'
        )
        
        nearby = []
        for node, dist in distances.items():
            if node != center_node and dist <= max_distance:
                desc = processor.descriptions.get(node, "No description")
                nearby.append(f"{node} ({desc}): {dist:.1f} units away")
        
        return "\n".join(nearby) if nearby else f"No nodes found within {max_distance} units of {center_node}"
    except Exception as e:
        return f"Error finding nearby nodes: {str(e)}"

def list_all_nodes() -> str:
    """List all nodes in the graph with their descriptions"""
    nodes_info = []
    for node_id, desc in processor.descriptions.items():
        connections = list(processor.graph.neighbors(node_id))
        nodes_info.append(f"{node_id}: {desc} (connected to: {connections})")
    return "\n".join(nodes_info) if nodes_info else "No nodes in graph"

def dic2LLMMessage(messages):
    autogen_messages: List[LLMMessage] = []
    for msg_dict in messages:
        role = msg_dict['role']
        content = msg_dict['content']

        if role == 'system':
            autogen_messages.append(SystemMessage(content=content))
        elif role == 'user':
            # You need to provide a 'source' for UserMessage.
            # I'm using "user_proxy" as a common default in AutoGen, but adjust as needed.
            autogen_messages.append(UserMessage(content=content, source="user_proxy"))
        elif role == 'assistant':
            # You need to provide a 'source' for AssistantMessage.
            # I'm using "assistant" as a common default, but adjust as needed.
            # Also, check if content is a string or a list of FunctionCall for AssistantMessage
            autogen_messages.append(AssistantMessage(content=content, source="assistant"))
        else:
            print(f"Warning: Unhandled message role encountered: {role}. Skipping this message.")
    return autogen_messages


class ImageGraphAgent:
    def __init__(self):
        self.model_client = OllamaChatCompletionClient(
            model="llama3.1:8b",
            base_url="http://localhost:11434",  # Replace with your actual Ollama base URL
            timeout=60
        )
        
        self.tools = {
            "search_image_nodes": {
                "function": search_image_nodes,
                "description": "Search for image nodes matching a text description. Args: query (str)"
            },
            "find_shortest_path": {
                "function": find_shortest_path,
                "description": "Find the shortest path between two nodes. Args: start_node (str), end_node (str)"
            },
            "find_nearby_nodes": {
                "function": find_nearby_nodes,
                "description": "Find nodes within a specified distance of a center node. Args: center_node (str), max_distance (float)"
            },
            "list_all_nodes": {
                "function": list_all_nodes,
                "description": "List all nodes and their connections. No arguments needed."
            }
        }

    def _call_ollama_sync(self, messages: List[Dict]) -> str:
        async def _async_call():
            response = await self.model_client.create(messages)
            return response.content if hasattr(response, 'content') else str(response)

        try:
            return asyncio.run(_async_call())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_async_call())

    def get_tools_description(self) -> str:
        return "\n".join([f"{name}: {info['description']}" for name, info in self.tools.items()])

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        if tool_name in self.tools:
            try:
                return self.tools[tool_name]
            except Exception as e:
                return f"Error executing {tool_name}: {str(e)}"
        return f"Tool {tool_name} not found."

    def parse_and_execute_tools(self, response: str) -> str:
        tool_pattern = r'\[(\w+)\((.*?)\)\]'
        matches = re.findall(tool_pattern, response)
        results = []

        for tool_name, args_str in matches:
            kwargs = {}
            if args_str.strip():
                for arg in args_str.split(','):
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        key = key.strip().strip('"\'')
                        value = value.strip().strip('"\'')
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                        kwargs[key] = value
            result = self.execute_tool(tool_name, **kwargs)
            results.append(f"Tool {tool_name} result:\n{result}")
        return "\n\n".join(results) if results else ""

    def answer_question(self, question: str) -> str:
        print(f"\n🤔 Question: {question}")
        system_message =f"""You are an expert assistant that helps users navigate and analyze image graphs.

                    The graph consists of:
                    - Nodes: Images with text descriptions
                    - Edges: Distances between images in some unit (meters, pixels, etc.)

                    {self.get_tools_description()}

                    When you need to use a tool, format it as: [TOOL_NAME(arg1="value1", arg2="value2")]

                    For example:
                    - To search for nodes: [search_image_nodes(query="red car")]
                    - To find a path: [find_shortest_path(start_node="img1", end_node="img2")]
                    - To find nearby nodes: [find_nearby_nodes(center_node="img1", max_distance=500)]
                    - To list all nodes: [list_all_nodes()]

                    Always use the tools to get accurate information before providing your final answer."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]

        initial_response = self._call_ollama_sync(dic2LLMMessage(messages))
        print(f"🤖 Initial response: {initial_response}")

        tool_results = self.parse_and_execute_tools(initial_response)
        if tool_results:
            print(f"🔧 Tool results:\n{tool_results}")
            messages.extend([
                {"role": "assistant", "content": initial_response},
                {"role": "user", "content": f"Here are the tool results:\n\n{tool_results}\n\nPlease provide a comprehensive answer to the original question based on this information."}
            ])
            final_response = self._call_ollama_sync(dic2LLMMessage(messages))
            return final_response
        return initial_response

    def add_image_node(self, node_id: str, description: str, image_path: str = None):
        processor.add_image_node(node_id, description, image_path)

    def add_distance_edge(self, node1: str, node2: str, distance: float):
        processor.add_distance_edge(node1, node2, distance)

# Example usage
def main():
    agent = ImageGraphAgent()
    agent.add_image_node("car", "A red car in a parking lot")
    agent.add_image_node("park", "A green park with trees")
    agent.add_distance_edge("car", "park", 10.5)

    question = "What's the shortest path from the red car to the park?"
    print("\n" + "="*50)
    answer = agent.answer_question(question)
    print(f"\n📣 Final Answer:\n{answer}")

if __name__ == "__main__":
    main()
