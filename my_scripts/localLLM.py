
from smolagents import CodeAgent, DuckDuckGoSearchTool, tool
from smolagents.models import TransformersModel

from smolagents import load_tool, CodeAgent, DuckDuckGoSearchTool
from dotenv import load_dotenv
import ollama
from dataclasses import dataclass

# # Load environment variables
# load_dotenv()

# @dataclass
# class Message:
#     content: str  # Required attribute for smolagents

# class OllamaModel:
#     def __init__(self, model_name):
#         self.model_name = model_name
#         self.client = ollama.Client()

#     def generate(self, messages, **kwargs):
#         formatted_messages = []
        
#         # Ensure messages are correctly formatted
#         for msg in messages:
#             if isinstance(msg, str):
#                 formatted_messages.append({
#                     "role": "user",  # Default to 'user' for plain strings
#                     "content": msg
#                 })
#             elif isinstance(msg, dict):
#                 role = msg.get("role", "user")
#                 content = msg.get("content", "")
#                 if isinstance(content, list):
#                     content = " ".join(part.get("text", "") for part in content if isinstance(part, dict) and "text" in part)
#                 formatted_messages.append({
#                     "role": role if role in ['user', 'assistant', 'system', 'tool'] else 'user',
#                     "content": content
#                 })
#             else:
#                 formatted_messages.append({
#                     "role": "user",  # Default role for unexpected types
#                     "content": str(msg)
#                 })

#         response = self.client.chat(
#             model=self.model_name,
#             messages=formatted_messages,
#             options={'temperature': 0.7, 'stream': False}
#         )
        
#         # Return a Message object with the 'content' attribute
#         return Message(
#             content=response.get("message", {}).get("content", "")
#         )
@tool
def describe_image(image_path: str) -> str:
    """
    Describe the content of an image given its URL.
    
    Args:
        image_path (str): The path to the image to describe.
    Returns:
        str: A description of the image.
    """
    if not image_path:
        return "No image path provided."
    
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

    raw_image = Image.open(image_path).convert('RGB')

    question = "Describe the content of this image in detail"
    inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True).strip()



# Define tools
# image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
search_tool = DuckDuckGoSearchTool()

# # Define the custom Ollama model
# ollama_model = OllamaModel("mistral-small:24b-instruct-2501-q8_0")

# # Create the agent
# agent = CodeAgent(
#     tools=[search_tool, image_generation_tool],
#     model=ollama_model,
#     planning_interval=3
# )

# # Run the agent
# result = agent.run(
#     "generate an image of a futuristic city skyline at sunset, "
# )

# # Output the result
# print(result)

# Working 
# ===================================================
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

model = TransformersModel(model_id, device_map='cuda:2')
agent = CodeAgent(tools=[search_tool, describe_image], model=model, max_steps=3)
# result = agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
result = agent.run("/scratch3/kat049/segment-anything-2-real-time/test.png")
print(result)

# ===================================================






# model = VLLMModel(
#     model_id="mistralai/Mistral-7B-Instruct-v0.3",
#     model_kwargs={"revision": "main", "gpu_memory_utilization":0.6},
# ) #runs out of memory 

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from smolagents import CodeAgent, DuckDuckGoSearchTool, ChatMessage, MessageRole

# class MistralModel:
#     def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.3", device="cuda"):
#         self.model_id = model_id
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             torch_dtype=torch.bfloat16,
#             device_map="auto"
#         )
#         self.device = device

#     def generate(self, prompt, stop_sequences=None) -> str:
#         if isinstance(prompt, list):
#             chat_prompt = []
#             for msg in prompt:
#                 role = msg.get("role", "user")
#                 content = msg.get("content", "")
#                 # Flatten list of blocks (e.g., [{'text': 'abc'}, {'text': 'def'}])
#                 if isinstance(content, list):
#                     flattened = ""
#                     for block in content:
#                         if isinstance(block, dict) and "text" in block:
#                             flattened += block["text"]
#                         elif isinstance(block, str):
#                             flattened += block
#                     content = flattened
#                 chat_prompt.append({"role": role, "content": content})

#             inputs = self.tokenizer.apply_chat_template(
#                 chat_prompt,
#                 add_generation_prompt=True,
#                 return_dict=True,
#                 return_tensors="pt",
#             ).to(self.model.device)

#         else:
#             # Fallback for raw string prompts
#             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=512,
#             eos_token_id=self.tokenizer.eos_token_id,
#         )
#         text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

#         if stop_sequences:
#             for stop in stop_sequences:
#                 if stop in text:
#                     text = text.split(stop)[0]
#                     break

#         return ChatMessage(role=MessageRole.ASSISTANT, content=text)


# agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=MistralModel())
# response = agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
# print(response)

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from smolagents.models import ChatMessage, TokenUsage, VLLMModel, TransformersModel

# def normalize_messages(messages):
#     normalized = []
#     for m in messages:
#         role = m["role"].value if hasattr(m["role"], "value") else m["role"]
#         content = m["content"]
#         if isinstance(content, list):
#             parts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in content]
#             content = "".join(parts)
#         normalized.append({"role": role, "content": content})
#     return normalized

# class MistralSmolWrapper:
#     def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.3"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id, torch_dtype=torch.bfloat16, device_map="auto"
#         )

#     def generate(self, messages, stop_sequences=None, tools_to_call_from=None, **kwargs):
#         messages = normalize_messages(messages)

#         inputs = self.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_dict=True,
#             return_tensors="pt"
#         )

#         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

#         output = self.model.generate(
#             **inputs,
#             max_new_tokens=kwargs.get("max_new_tokens", 512),
#             do_sample=True,
#         )

#         decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
#         # crude way to extract just the response
#         reply = decoded.split(messages[-1]["content"])[-1].strip()

#         return ChatMessage(role="assistant", content=reply, raw={"text": decoded}, token_usage=TokenUsage(0, 0))


# from smolagents import CodeAgent, DuckDuckGoSearchTool

# wrapped_model = MistralSmolWrapper()

# model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# agent = CodeAgent(
#     tools=[DuckDuckGoSearchTool()],
#     model=VLLMModel(model_id),
# )

# agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")

