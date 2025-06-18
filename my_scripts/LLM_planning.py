import json
import re
import json
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from my_scripts.utils_graph import GraphNode, GraphState  # Assuming you have a utils_graph.py with GraphNode defined

class Tool(ABC):
    """Base class for agent tools"""
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass

class TrajectoryAnalysisTool(Tool):
    """Tool for analyzing object trajectories"""
    
    def __init__(self, graph_data: Dict[str, GraphState]):
        self.graph_data = graph_data
    
    @property
    def description(self) -> str:
        return """
        Analyzes the trajectory of an object across time.
        Parameters:
        - object_id: ID of the object to track
        - start_time: Starting timestamp (optional)
        - end_time: Ending timestamp (optional)
        
        Returns trajectory data including positions, movements, and distances.
        """
    
    def execute(self, object_id: str, start_time: str = None, end_time: str = None) -> Dict[str, Any]:
        timestamps = sorted(self.graph_data.keys())
        
        if start_time:
            timestamps = [t for t in timestamps if t >= start_time]
        if end_time:
            timestamps = [t for t in timestamps if t <= end_time]
        
        trajectory = []
        movements = []
        
        for i, timestamp in enumerate(timestamps):
            if object_id in self.graph_data[timestamp].nodes:
                node = self.graph_data[timestamp].nodes[object_id]
                trajectory.append({
                    'timestamp': timestamp,
                    'position': node.position,
                    'text': node.text
                })
                
                # Calculate movement from previous position
                if i > 0 and len(trajectory) > 1:
                    prev_pos = trajectory[-2]['position']
                    curr_pos = node.position
                    
                    displacement = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
                    distance = math.sqrt(displacement[0]**2 + displacement[1]**2)
                    direction = math.atan2(displacement[1], displacement[0])
                    
                    movements.append({
                        'from_time': trajectory[-2]['timestamp'],
                        'to_time': timestamp,
                        'displacement': displacement,
                        'distance': distance,
                        'direction_radians': direction,
                        'direction_degrees': math.degrees(direction)
                    })
        
        return {
            'object_id': object_id,
            'trajectory': trajectory,
            'movements': movements,
            'total_distance': sum(m['distance'] for m in movements),
            'found_in_timestamps': len(trajectory)
        }

class SpatialRelationshipTool(Tool):
    """Tool for analyzing spatial relationships between objects"""
    
    def __init__(self, graph_data: Dict[str, GraphState]):
        self.graph_data = graph_data
    
    @property
    def description(self) -> str:
        return """
        Analyzes spatial relationships between objects at a given timestamp.
        Parameters:
        - timestamp: The time point to analyze
        - object1_id: ID of first object
        - object2_id: ID of second object (optional, defaults to 'self')
        
        Returns distance, direction, and relative position information.
        """
    
    def execute(self, timestamp: int, object1_id: str, object2_id: str = 'self') -> Dict[str, Any]:
        if timestamp not in self.graph_data:
            return {'error': f'Timestamp {timestamp} not found'}
        
        nodes = self.graph_data[timestamp].nodes
        
        if object1_id not in nodes or object2_id not in nodes:
            return {'error': f'One or both objects not found at timestamp {timestamp}'}
        
        pos1 = nodes[object1_id].position
        pos2 = nodes[object2_id].position
        
        displacement = (pos1[0] - pos2[0], pos1[1] - pos2[1])
        distance = math.sqrt(displacement[0]**2 + displacement[1]**2)
        direction = math.atan2(displacement[1], displacement[0])
        
        return {
            'timestamp': timestamp,
            'object1': object1_id,
            'object2': object2_id,
            'distance': distance,
            'displacement': displacement,
            'direction_radians': direction,
            'direction_degrees': math.degrees(direction),
            'direction_description': self._get_direction_description(direction)
        }
    
    def _get_direction_description(self, angle_radians: float) -> str:
        angle_degrees = math.degrees(angle_radians)
        
        if -22.5 <= angle_degrees <= 22.5:
            return "to the right"
        elif 22.5 < angle_degrees <= 67.5:
            return "towards upper-right"
        elif 67.5 < angle_degrees <= 112.5:
            return "upward"
        elif 112.5 < angle_degrees <= 157.5:
            return "towards upper-left"
        elif 157.5 < angle_degrees or angle_degrees <= -157.5:
            return "to the left"
        elif -157.5 < angle_degrees <= -112.5:
            return "towards lower-left"
        elif -112.5 < angle_degrees <= -67.5:
            return "downward"
        else:
            return "towards lower-right"

class VisualAnalysisTool(Tool):
    """Tool for analyzing visual changes using your Aria model"""
    
    def __init__(self, graph_data: Dict[str, GraphState], aria_function):
        self.graph_data = graph_data
        self.aria_function = aria_function
    
    @property
    def description(self) -> str:
        return """
        Analyzes visual changes of an object between timestamps using Aria model.
        Parameters:
        - object_id: ID of the object to analyze
        - timestamp1: First timestamp
        - timestamp2: Second timestamp
        - question: Specific question about visual changes (optional)
        
        Returns visual analysis description.
        """
    
    def execute(self, object_id: str, timestamp1: str, timestamp2: str, 
                question: str = "How has this object changed visually?") -> Dict[str, Any]:
        
        if timestamp1 not in self.graph_data or timestamp2 not in self.graph_data:
            return {'error': 'One or both timestamps not found'}
        
        nodes1 = self.graph_data[timestamp1].nodes
        nodes2 = self.graph_data[timestamp2].nodes
        
        if object_id not in nodes1 or object_id not in nodes2:
            return {'error': f'Object {object_id} not found in one or both timestamps'}
        
        image1 = nodes1[object_id].image
        image2 = nodes2[object_id].image
        
        # Use your Aria function for visual analysis
        visual_analysis = self.aria_function(question, [image1, image2])
        
        return {
            'object_id': object_id,
            'timestamp1': timestamp1,
            'timestamp2': timestamp2,
            'question': question,
            'visual_analysis': visual_analysis
        }

class TextAnalysisTool(Tool):
    """Tool for analyzing textual changes"""
    
    def __init__(self, graph_data: Dict[str, GraphState]):
        self.graph_data = graph_data
    
    @property
    def description(self) -> str:
        return """
        Analyzes textual changes of an object between timestamps.
        Parameters:
        - object_id: ID of the object to analyze
        - timestamp1: First timestamp
        - timestamp2: Second timestamp
        
        Returns text comparison and changes.
        """
    
    def execute(self, object_id: str, timestamp1: str, timestamp2: str) -> Dict[str, Any]:
        if timestamp1 not in self.graph_data or timestamp2 not in self.graph_data:
            return {'error': 'One or both timestamps not found'}
        
        nodes1 = self.graph_data[timestamp1].nodes
        nodes2 = self.graph_data[timestamp2].nodes
        
        if object_id not in nodes1 or object_id not in nodes2:
            return {'error': f'Object {object_id} not found in one or both timestamps'}
        
        text1 = nodes1[object_id].text
        text2 = nodes2[object_id].text
        
        return {
            'object_id': object_id,
            'timestamp1': timestamp1,
            'timestamp2': timestamp2,
            'text1': text1,
            'text2': text2,
            'changed': text1 != text2,
            'changes': self._analyze_text_changes(text1, text2)
        }
    
    def _analyze_text_changes(self, text1: str, text2: str) -> Dict[str, Any]:
        # Simple text analysis - you could use more sophisticated NLP here
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        return {
            'added_words': list(words2 - words1),
            'removed_words': list(words1 - words2),
            'common_words': list(words1 & words2),
            'similarity_score': len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
        }

class GraphQueryTool(Tool):
    """Tool for general graph queries"""
    
    def __init__(self, graph_data: Dict[str, GraphState]):
        self.graph_data = graph_data
    
    @property
    def description(self) -> str:
        return """
        Performs general queries on the graph data.
        Parameters:
        - query_type: 'list_objects', 'list_timestamps', 'object_exists', 'get_object_info'
        - timestamp: Specific timestamp (for relevant queries)
        - object_id: Specific object ID (for relevant queries)
        
        Returns requested information about the graph.
        """
    
    def execute(self, query_type: str, timestamp: str = None, object_id: str = None) -> Dict[str, Any]:
        if query_type == 'list_timestamps':
            return {'timestamps': sorted(self.graph_data.keys())}
        
        elif query_type == 'list_objects' and timestamp:
            if timestamp in self.graph_data:
                return {'objects': list(self.graph_data[timestamp].nodes.keys())}
            return {'error': f'Timestamp {timestamp} not found'}
        
        elif query_type == 'object_exists' and timestamp and object_id:
            exists = timestamp in self.graph_data and object_id in self.graph_data[timestamp].nodes
            return {'exists': exists}
        
        elif query_type == 'get_object_info' and timestamp and object_id:
            if timestamp in self.graph_data and object_id in self.graph_data[timestamp].nodes:
                node = self.graph_data[timestamp].nodes[object_id]
                return {
                    'object_id': object_id,
                    'timestamp': timestamp,
                    'position': node.position,
                    'text': node.text
                }
            return {'error': f'Object {object_id} not found at timestamp {timestamp}'}
        
        return {'error': 'Invalid query parameters'}

class SpatioTemporalAgent:
    """Main agent for spatio-temporal reasoning"""
    
    def __init__(self, graph_data: Dict[str, GraphState], aria_function=None):
        self.graph_data = graph_data
        self.tools = {
            'trajectory_analysis': TrajectoryAnalysisTool(graph_data),
            'spatial_relationship': SpatialRelationshipTool(graph_data),
            'visual_analysis': VisualAnalysisTool(graph_data, aria_function) if aria_function else None,
            'text_analysis': TextAnalysisTool(graph_data),
            'graph_query': GraphQueryTool(graph_data)
        }
    
    def get_available_tools(self) -> Dict[str, str]:
        """Get descriptions of all available tools"""
        return {name: tool.description for name, tool in self.tools.items() if tool is not None}
    
    def use_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific tool"""
        if tool_name not in self.tools or self.tools[tool_name] is None:
            return {'error': f'Tool {tool_name} not available'}
        
        try:
            return self.tools[tool_name].execute(**kwargs)
        except Exception as e:
            return {'error': f'Tool execution failed: {str(e)}'}
    
    def answer_movement_question(self, object_id: str, reference_object: str = 'self') -> str:
        """Answer 'How did object X move relative to Y?' using tools"""
        
        # Step 1: Get trajectory of the object
        trajectory_result = self.use_tool('trajectory_analysis', object_id=object_id)
        
        if 'error' in trajectory_result:
            return f"Could not analyze trajectory: {trajectory_result['error']}"
        
        if trajectory_result['found_in_timestamps'] < 2:
            return f"Insufficient data to track movement of {object_id}"
        
        # Step 2: Analyze movements
        movements = trajectory_result['movements']
        total_distance = trajectory_result['total_distance']
        
        # Step 3: Generate comprehensive answer
        answer = f"Movement analysis for {object_id} relative to {reference_object}:\n\n"
        answer += f"Total distance traveled: {total_distance:.2f} units\n"
        answer += f"Number of time steps: {len(movements)}\n\n"
        
        for movement in movements:
            dx, dy = movement['displacement']
            distance = movement['distance']
            direction = movement.get('direction_description', 'unknown direction')
            
            answer += f"From {movement['from_time']} to {movement['to_time']}:\n"
            answer += f"- Moved {distance:.2f} units {direction}\n"
            answer += f"- Displacement: ({dx:+.1f}, {dy:+.1f})\n"
            
            # Add visual analysis if available
            if self.tools['visual_analysis']:
                visual_result = self.use_tool('visual_analysis', 
                                            object_id=object_id,
                                            timestamp1=movement['from_time'],
                                            timestamp2=movement['to_time'])
                if 'visual_analysis' in visual_result:
                    answer += f"- Visual change: {visual_result['visual_analysis']}\n"
            
            # Add text analysis
            text_result = self.use_tool('text_analysis',
                                      object_id=object_id,
                                      timestamp1=movement['from_time'],
                                      timestamp2=movement['to_time'])
            if 'changed' in text_result and text_result['changed']:
                answer += f"- Text description changed\n"
            
            answer += "\n"
        
        return answer

# Example usage and integration
def create_sample_data():
    """Create sample graph data for testing"""
    sample_data = {}
    
    # Create sample nodes for different timestamps
    for i, timestamp in enumerate(['t1', 't2', 't3']):
        nodes = {
            'self': GraphNode('self', 'text', 'Observer', 'Observer position', timestamp, (0, 0)),
            'object_A': GraphNode('object_A', 'mask', None, f'Red car at {timestamp}', timestamp, (5 + i, 3 + i*0.5)),
            'object_B': GraphNode('object_B', 'mask', None, f'Blue building at {timestamp}', timestamp, (-2 + i*0.3, 4 - i*0.2))
        }
        sample_data[timestamp] = GraphState(nodes, timestamp)
    
    return sample_data

# # Usage example
# if __name__ == "__main__":
#     # Create sample data
#     graph_data = create_sample_data()
    
#     # Initialize agent (without Aria for this example)
#     agent = SpatioTemporalAgent(graph_data)
    
#     # Get available tools
#     print("Available tools:")
#     for tool_name, description in agent.get_available_tools().items():
#         print(f"- {tool_name}: {description.strip()}")
    
#     print("\n" + "="*50 + "\n")
    
#     # Answer movement question
#     answer = agent.answer_movement_question('object_A')
#     print(answer)
    
#     # Use individual tools
#     print("="*50)
#     print("Individual tool usage:")
    
#     # Check spatial relationship
#     spatial_result = agent.use_tool('spatial_relationship', 
#                                    timestamp='t2', 
#                                    object1_id='object_A', 
#                                    object2_id='self')
#     print(f"Spatial relationship at t2: {spatial_result}")
    
#     # Get trajectory
#     trajectory_result = agent.use_tool('trajectory_analysis', object_id='object_A')
#     print(f"Trajectory summary: {trajectory_result['total_distance']:.2f} units total distance")

class LLMPlanningAgent:
    """Agent that uses LLM for planning and tool selection"""
    
    def __init__(self, spatiotemporal_agent: SpatioTemporalAgent, llm_function=None):
        self.st_agent = spatiotemporal_agent
        self.llm_function = llm_function  # Your LLM function (e.g., OpenAI, Claude, etc.)
        self.conversation_history = []
    
    def answer_question(self, question: str) -> str:
        """Main entry point for answering questions"""
        
        # Step 1: Plan the approach using LLM
        plan = self._create_plan(question)
        
        # Step 2: Execute the plan
        results = self._execute_plan(plan)
        
        # Step 3: Synthesize final answer
        final_answer = self._synthesize_answer(question, results)
        
        # Store in conversation history
        self.conversation_history.append({
            'question': question,
            'plan': plan,
            'results': results,
            'answer': final_answer
        })
        
        return final_answer
    
    def _create_plan(self, question: str) -> List[Dict[str, Any]]:
        """Use LLM to create a plan for answering the question"""
        
        available_tools = self.st_agent.get_available_tools()
        tools_description = "\n".join([f"- {name}: {desc}" for name, desc in available_tools.items()])
        
        planning_prompt = f"""
        You are a spatio-temporal reasoning assistant. A user has asked the following question:
        
        Question: "{question}"
        
        You have access to these tools:
        {tools_description}
        
        Create a step-by-step plan to answer this question. Return your plan as a JSON list where each step has:
        - "step": step number
        - "action": "use_tool" or "analyze" or "synthesize"
        - "tool_name": name of tool to use (if action is "use_tool")
        - "parameters": parameters for the tool (if action is "use_tool")
        - "purpose": what this step aims to accomplish
        
        Example plan format:
        [
            {{
                "step": 1,
                "action": "use_tool",
                "tool_name": "trajectory_analysis",
                "parameters": {{"object_id": "object_A"}},
                "purpose": "Get movement data for object A"
            }},
            {{
                "step": 2,
                "action": "analyze",
                "purpose": "Interpret the trajectory data"
            }}
        ]
        
        Focus on being efficient and only using tools that are necessary to answer the question.
        """
        
        if self.llm_function:
            response = self.llm_function(planning_prompt)
            try:
                # Extract JSON from response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    plan = json.loads(json_match.group())
                    return plan
            except:
                pass
        
        # Fallback: create a simple plan based on question keywords
        return self._create_simple_plan(question)
    
    def _create_simple_plan(self, question: str) -> List[Dict[str, Any]]:
        # TODO get rid of
        """Create a simple plan based on question keywords"""
        question_lower = question.lower()
        plan = []
        
        if "move" in question_lower or "trajectory" in question_lower:
            # Extract object name from question
            object_match = re.search(r'object[_\s]([a-zA-Z0-9]+)', question_lower)
            object_id = object_match.group(1) if object_match else "A"
            
            plan.append({
                "step": 1,
                "action": "use_tool",
                "tool_name": "trajectory_analysis",
                "parameters": {"object_id": f"object_{object_id}"},
                "purpose": f"Analyze trajectory of object_{object_id}"
            })
            
        if "visual" in question_lower or "appear" in question_lower or "look" in question_lower:
            plan.append({
                "step": len(plan) + 1,
                "action": "use_tool",
                "tool_name": "visual_analysis",
                "parameters": {"object_id": "object_A", "timestamp1": "t1", "timestamp2": "t2"},
                "purpose": "Analyze visual changes"
            })
        
        if "relation" in question_lower or "distance" in question_lower or "between" in question_lower:
            # plan.append({
            #     "step": len(plan) + 1,
            #     "action": "use_tool",
            #     "tool_name": "spatial_relationship",
            #     "parameters": {"timestamp": "t1", "object1_id": "object_A"},
            #     "purpose": "Analyze spatial relationships"
            # })
            objs = re.findall(r'mask\s*(\d+)', question.lower())
            plan.append({
                "step": len(plan) + 1,
                "action": "use_tool",
                "tool_name": "spatial_relationship",
                "parameters": {"timestamp": int(re.search(r' (\d+)', question.lower()).group(1)), "object1_id": f'mask{objs[0]}' if objs else 'self', "object2_id": f'mask{objs[1]}' if len(objs) > 1 else 'self'},
                "purpose": "Analyze spatial relationships"
            })
        
        if not plan:  # Default plan
            plan.append({
                "step": 1,
                "action": "use_tool",
                "tool_name": "graph_query",
                "parameters": {"query_type": "list_timestamps"},
                "purpose": "Get overview of available data"
            })
        
        return plan
    
    def _execute_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute the planned steps"""
        results = []
        
        for step in plan:
            if step["action"] == "use_tool":
                tool_result = self.st_agent.use_tool(
                    step["tool_name"], 
                    **step["parameters"]
                )
                results.append({
                    "step": step["step"],
                    "purpose": step["purpose"],
                    "result": tool_result
                })
            elif step["action"] == "analyze":
                # For now, just record that analysis was requested
                results.append({
                    "step": step["step"],
                    "purpose": step["purpose"],
                    "result": {"action": "analysis_placeholder"}
                })
        
        return results
    
    def _synthesize_answer(self, question: str, results: List[Dict[str, Any]]) -> str:
        """Synthesize final answer from tool results"""
        
        if not results:
            return "I couldn't gather enough information to answer your question."
        
        # Prepare synthesis prompt
        results_summary = ""
        for result in results:
            results_summary += f"Step {result['step']} ({result['purpose']}):\n"
            results_summary += f"Result: {json.dumps(result['result'], indent=2)}\n\n"
        
        synthesis_prompt = f"""
        Based on the following analysis results, provide a clear and comprehensive answer to the user's question.
        
        Original question: "{question}"
        
        Analysis results:
        {results_summary}
        
        Provide a natural language answer that directly addresses the user's question. Be specific and include relevant details from the analysis.
        """
        
        if self.llm_function:
            return self.llm_function(synthesis_prompt)
        else:
            # Fallback: simple synthesis
            return self._simple_synthesis(question, results)
    
    def _simple_synthesis(self, question: str, results: List[Dict[str, Any]]) -> str:
        """Simple synthesis without LLM"""
        answer = f"Based on the analysis:\n\n"
        
        for result in results:
            answer += f"• {result['purpose']}: "
            if 'error' in result['result']:
                answer += f"Error - {result['result']['error']}\n"
            elif 'trajectory' in result['result']:
                traj = result['result']
                answer += f"Object moved {traj['total_distance']:.2f} units across {traj['found_in_timestamps']} timestamps\n"
            elif 'distance' in result['result']:
                spatial = result['result']
                answer += f"Distance of {spatial['distance']:.2f} units {spatial.get('direction_description', '')}\n"
            else:
                answer += "Analysis completed\n"
        
        return answer

class ConversationalAgent:
    """High-level conversational interface"""
    
    def __init__(self, graph_data: Dict[str, GraphState], aria_function=None, llm_function=None):
        self.st_agent = SpatioTemporalAgent(graph_data, aria_function)
        self.planning_agent = LLMPlanningAgent(self.st_agent, llm_function)
        self.context = {
            'current_focus_object': None,
            'current_timeframe': None,
            'recent_queries': []
        }
    
    def chat(self, message: str) -> str:
        """Main chat interface"""
        
        # Update context based on message
        self._update_context(message)
        
        # Handle different types of queries
        if self._is_movement_question(message):
            return self._handle_movement_question(message)
        elif self._is_comparison_question(message):
            return self._handle_comparison_question(message)
        elif self._is_general_question(message):
            return self.planning_agent.answer_question(message)
        else:
            return self._handle_general_conversation(message)
    
    def _update_context(self, message: str):
        """Update conversation context"""
        # Extract object references
        object_match = re.search(r'mask\s*(\d+)', message.lower())         
        if object_match:
            self.context['current_focus_object'] = f"mask{object_match.group(1)}"
        
        # Extract time references
        time_match = re.search(r' (\d+)', message.lower())
        if time_match:
            self.context['current_timeframe'] = int(time_match.group(1))
        
        # Store recent query
        self.context['recent_queries'].append(message)
        if len(self.context['recent_queries']) > 5:
            self.context['recent_queries'].pop(0)
    
    def _is_movement_question(self, message: str) -> bool:
        movement_keywords = ['move', 'moved', 'trajectory', 'path', 'travel', 'motion']
        return any(keyword in message.lower() for keyword in movement_keywords)
    
    def _is_comparison_question(self, message: str) -> bool:
        comparison_keywords = ['compare', 'difference', 'between', 'versus', 'vs']
        return any(keyword in message.lower() for keyword in comparison_keywords)
    
    def _is_general_question(self, message: str) -> bool:
        question_indicators = ['?', 'what', 'how', 'where', 'when', 'why', 'which']
        return any(indicator in message.lower() for indicator in question_indicators)
    
    def _handle_movement_question(self, message: str) -> str:
        """Handle movement-specific questions with optimized approach"""
        
        # Extract object from message or use context
        object_match =  re.search(r'mask\s*(\d+)', message.lower())
        if object_match:
            object_id = f"mask{object_match.group(1)}"
        elif self.context['current_focus_object']:
            object_id = self.context['current_focus_object']
        else:
            object_id = 'mask0'  # Default
        
        return self.st_agent.answer_movement_question(object_id)
    
    def _handle_comparison_question(self, message: str) -> str:
        """Handle comparison questions"""
        return self.planning_agent.answer_question(message)
    
    def _handle_general_conversation(self, message: str) -> str:
        """Handle general conversation"""
        return f"I understand you're asking about: '{message}'. I can help you analyze spatio-temporal relationships in your graph data. Try asking about object movements, spatial relationships, or visual changes!"

# Integration with your existing Aria function
def integrate_with_aria(answerwAria_function, graph_data):
    """Integration example with your existing Aria function"""
    
    # Create the conversational agent
    agent = ConversationalAgent(
        graph_data=graph_data,
        aria_function=answerwAria_function,
        llm_function=None  # You can plug in your preferred LLM here
    )
    
    return agent

# Example usage
# def demo_conversation():
#     """Demo conversation with the agent"""
    
#     # Create sample data
#     sample_data = create_sample_data()  # From previous artifact
    
#     # Initialize agent
#     agent = ConversationalAgent(sample_data)
    
#     # Example conversation
#     questions = [
#         "How did object A move relative to self?",
#         "What's the distance between object A and object B at t2?",
#         "Compare the positions of object A at t1 and t3",
#         "Tell me about the trajectory of object B"
#     ]
    
#     print("=== Demo Conversation ===\n")
    
#     for i, question in enumerate(questions, 1):
#         print(f"Q{i}: {question}")
#         answer = agent.chat(question)
#         print(f"A{i}: {answer}\n" + "-"*50 + "\n")

# Advanced tool orchestration with chain-of-thought
class ChainOfThoughtAgent:
    """Agent that performs chain-of-thought reasoning with tools"""
    
    def __init__(self, spatiotemporal_agent: SpatioTemporalAgent, llm_function=None):
        self.st_agent = spatiotemporal_agent
        self.llm_function = llm_function
        self.reasoning_steps = []
    
    def solve_complex_query(self, query: str) -> Dict[str, Any]:
        """Solve complex spatio-temporal queries with step-by-step reasoning"""
        
        self.reasoning_steps = []
        
        # Step 1: Break down the query
        decomposition = self._decompose_query(query)
        
        # Step 2: Execute sub-queries
        sub_results = []
        for sub_query in decomposition['sub_queries']:
            result = self._execute_sub_query(sub_query)
            sub_results.append(result)
        
        # Step 3: Synthesize results
        final_result = self._synthesize_complex_result(query, sub_results)
        
        return {
            'original_query': query,
            'decomposition': decomposition,
            'sub_results': sub_results,
            'final_result': final_result,
            'reasoning_steps': self.reasoning_steps
        }
    
    def _decompose_query(self, query: str) -> Dict[str, Any]:
        """Decompose complex query into sub-queries"""
        
        # Example decompositions based on query patterns
        query_lower = query.lower()
        
        if "fastest" in query_lower or "slowest" in query_lower:
            return {
                'query_type': 'speed_comparison',
                'sub_queries': [
                    {'type': 'get_all_objects', 'params': {}},
                    {'type': 'calculate_speeds', 'params': {}},
                    {'type': 'compare_speeds', 'params': {}}
                ]
            }
        
        elif "closest" in query_lower or "farthest" in query_lower:
            return {
                'query_type': 'distance_comparison',
                'sub_queries': [
                    {'type': 'get_all_distances', 'params': {}},
                    {'type': 'find_extremes', 'params': {}}
                ]
            }
        
        elif "changed most" in query_lower or "changed least" in query_lower:
            return {
                'query_type': 'change_analysis',
                'sub_queries': [
                    {'type': 'analyze_all_changes', 'params': {}},
                    {'type': 'rank_changes', 'params': {}}
                ]
            }
        
        else:
            return {
                'query_type': 'general',
                'sub_queries': [
                    {'type': 'trajectory_analysis', 'params': {}},
                    {'type': 'spatial_analysis', 'params': {}}
                ]
            }
    
    def _execute_sub_query(self, sub_query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a sub-query using appropriate tools"""
        
        query_type = sub_query['type']
        self.reasoning_steps.append(f"Executing sub-query: {query_type}")
        
        if query_type == 'get_all_objects':
            # Get all objects across all timestamps
            timestamps = self.st_agent.use_tool('graph_query', query_type='list_timestamps')
            all_objects = set()
            
            for timestamp in timestamps.get('timestamps', []):
                objects = self.st_agent.use_tool('graph_query', 
                                               query_type='list_objects', 
                                               timestamp=timestamp)
                if 'objects' in objects:
                    all_objects.update(objects['objects'])
            
            return {'all_objects': list(all_objects)}
        
        elif query_type == 'calculate_speeds':
            # Calculate speeds for all objects
            all_objects_result = self._execute_sub_query({'type': 'get_all_objects', 'params': {}})
            speeds = {}
            
            for obj_id in all_objects_result['all_objects']:
                if obj_id != 'self':  # Skip self
                    trajectory = self.st_agent.use_tool('trajectory_analysis', object_id=obj_id)
                    if 'movements' in trajectory:
                        total_time = len(trajectory['movements'])
                        total_distance = trajectory['total_distance']
                        speeds[obj_id] = total_distance / max(total_time, 1)
            
            return {'speeds': speeds}
        
        elif query_type == 'get_all_distances':
            # Get distances between all objects and self at latest timestamp
            timestamps = self.st_agent.use_tool('graph_query', query_type='list_timestamps')
            latest_timestamp = max(timestamps.get('timestamps', ['t1']))
            
            objects = self.st_agent.use_tool('graph_query', 
                                           query_type='list_objects', 
                                           timestamp=latest_timestamp)
            distances = {}
            
            for obj_id in objects.get('objects', []):
                if obj_id != 'self':
                    spatial = self.st_agent.use_tool('spatial_relationship',
                                                   timestamp=latest_timestamp,
                                                   object1_id=obj_id,
                                                   object2_id='self')
                    if 'distance' in spatial:
                        distances[obj_id] = spatial['distance']
            
            return {'distances': distances, 'timestamp': latest_timestamp}
        
        elif query_type == 'analyze_all_changes':
            # Analyze changes for all objects
            timestamps = self.st_agent.use_tool('graph_query', query_type='list_timestamps')['timestamps']
            changes = {}
            
            if len(timestamps) >= 2:
                first_t = timestamps[0]
                last_t = timestamps[-1]
                
                objects_first = self.st_agent.use_tool('graph_query', 
                                                     query_type='list_objects', 
                                                     timestamp=first_t)
                
                for obj_id in objects_first.get('objects', []):
                    if obj_id != 'self':
                        trajectory = self.st_agent.use_tool('trajectory_analysis', object_id=obj_id)
                        if 'total_distance' in trajectory:
                            changes[obj_id] = trajectory['total_distance']
            
            return {'changes': changes}
        
        return {'result': 'sub_query_executed'}
    
    def _synthesize_complex_result(self, original_query: str, sub_results: List[Dict[str, Any]]) -> str:
        """Synthesize final result from sub-query results"""
        
        query_lower = original_query.lower()
        
        # Find speed-related results
        speeds_result = next((r for r in sub_results if 'speeds' in r), None)
        if speeds_result and ("fastest" in query_lower or "slowest" in query_lower):
            speeds = speeds_result['speeds']
            if speeds:
                if "fastest" in query_lower:
                    fastest_obj = max(speeds.items(), key=lambda x: x[1])
                    return f"The fastest moving object is {fastest_obj[0]} with a speed of {fastest_obj[1]:.2f} units per time step."
                else:
                    slowest_obj = min(speeds.items(), key=lambda x: x[1])
                    return f"The slowest moving object is {slowest_obj[0]} with a speed of {slowest_obj[1]:.2f} units per time step."
        
        # Find distance-related results
        distances_result = next((r for r in sub_results if 'distances' in r), None)
        if distances_result and ("closest" in query_lower or "farthest" in query_lower):
            distances = distances_result['distances']
            timestamp = distances_result['timestamp']
            if distances:
                if "closest" in query_lower:
                    closest_obj = min(distances.items(), key=lambda x: x[1])
                    return f"The closest object to self at {timestamp} is {closest_obj[0]} at a distance of {closest_obj[1]:.2f} units."
                else:
                    farthest_obj = max(distances.items(), key=lambda x: x[1])
                    return f"The farthest object from self at {timestamp} is {farthest_obj[0]} at a distance of {farthest_obj[1]:.2f} units."
        
        # Find change-related results
        changes_result = next((r for r in sub_results if 'changes' in r), None)
        if changes_result and ("changed most" in query_lower or "changed least" in query_lower):
            changes = changes_result['changes']
            if changes:
                if "changed most" in query_lower or "most" in query_lower:
                    most_changed = max(changes.items(), key=lambda x: x[1])
                    return f"The object that changed most is {most_changed[0]} with a total movement of {most_changed[1]:.2f} units."
                else:
                    least_changed = min(changes.items(), key=lambda x: x[1])
                    return f"The object that changed least is {least_changed[0]} with a total movement of {least_changed[1]:.2f} units."
        
        return "Analysis completed, but couldn't determine a specific answer for your query."

# Multi-agent system for collaborative reasoning
class MultiAgentSystem:
    """System that coordinates multiple specialized agents"""
    
    def __init__(self, graph_data: Dict[str, GraphState], aria_function=None):
        self.graph_data = graph_data
        
        # Initialize specialized agents
        self.movement_agent = SpatioTemporalAgent(graph_data, aria_function)
        self.visual_agent = SpatioTemporalAgent(graph_data, aria_function)  # Specialized for visual tasks
        self.planning_agent = LLMPlanningAgent(self.movement_agent)
        self.cot_agent = ChainOfThoughtAgent(self.movement_agent)
        
        # Agent capabilities
        self.agent_capabilities = {
            'movement_agent': ['trajectory', 'movement', 'speed', 'path'],
            'visual_agent': ['visual', 'appearance', 'color', 'shape', 'size'],
            'planning_agent': ['complex', 'multi-step', 'planning'],
            'cot_agent': ['comparison', 'ranking', 'analysis', 'fastest', 'closest']
        }
    
    def route_query(self, query: str) -> str:
        """Route query to the most appropriate agent"""
        
        query_lower = query.lower()
        scores = {}
        
        # Score each agent based on keyword matches
        for agent_name, keywords in self.agent_capabilities.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            scores[agent_name] = score
        
        # Select best agent
        best_agent = max(scores.items(), key=lambda x: x[1])[0]
        
        # Route to appropriate agent
        if best_agent == 'movement_agent':
            return self._use_movement_agent(query)
        elif best_agent == 'visual_agent':
            return self._use_visual_agent(query)
        elif best_agent == 'cot_agent':
            result = self.cot_agent.solve_complex_query(query)
            return result['final_result']
        else:  # planning_agent
            return self.planning_agent.answer_question(query)
    
    def _use_movement_agent(self, query: str) -> str:
        """Use movement agent for trajectory analysis"""
        # Extract object from query
        object_match = re.search(r'object[_\s]([a-zA-Z0-9]+)', query.lower())
        object_id = f"object_{object_match.group(1)}" if object_match else 'object_A'
        
        return self.movement_agent.answer_movement_question(object_id)
    
    def _use_visual_agent(self, query: str) -> str:
        """Use visual agent for appearance analysis"""
        if self.visual_agent.tools['visual_analysis']:
            # Get available timestamps
            timestamps = self.movement_agent.use_tool('graph_query', query_type='list_timestamps')['timestamps']
            
            if len(timestamps) >= 2:
                object_match = re.search(r'object[_\s]([a-zA-Z0-9]+)', query.lower())
                object_id = f"object_{object_match.group(1)}" if object_match else 'object_A'
                
                visual_result = self.visual_agent.use_tool('visual_analysis',
                                                         object_id=object_id,
                                                         timestamp1=timestamps[0],
                                                         timestamp2=timestamps[-1],
                                                         question=query)
                
                if 'visual_analysis' in visual_result:
                    return f"Visual analysis of {object_id}: {visual_result['visual_analysis']}"
        
        return "Visual analysis not available or insufficient data."

# Complete integration example
def create_complete_system(graph_data, aria_function, llm_function=None):
    """Create a complete multi-agent reasoning system"""
    
    return {
        'basic_agent': SpatioTemporalAgent(graph_data, aria_function),
        'conversational_agent': ConversationalAgent(graph_data, aria_function, llm_function),
        'planning_agent': LLMPlanningAgent(SpatioTemporalAgent(graph_data, aria_function), llm_function),
        'cot_agent': ChainOfThoughtAgent(SpatioTemporalAgent(graph_data, aria_function), llm_function),
        'multi_agent': MultiAgentSystem(graph_data, aria_function)
    }

#Usage example
if __name__ == "__main__":
    # Run the demo
    # demo_conversation()
    
    print("\n=== Chain of Thought Demo ===\n")
    
    # Demo chain of thought reasoning
    sample_data = create_sample_data()
    cot_agent = ChainOfThoughtAgent(SpatioTemporalAgent(sample_data))
    
    complex_queries = [
        "Which object moved the fastest?",
        "What object is closest to self?",
        "Which object changed the most?"
    ]
    
    for query in complex_queries:
        print(f"Query: {query}")
        result = cot_agent.solve_complex_query(query)
        print(f"Answer: {result['final_result']}")
        print(f"Reasoning steps: {', '.join(result['reasoning_steps'])}")
        print("-" * 50)