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


from smolagents import CodeAgent, DuckDuckGoSearchTool

# wrapped_model = MistralSmolWrapper()

# model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# agent = CodeAgent(
#     tools=[DuckDuckGoSearchTool()],
#     model=VLLMModel(model_id),
# )

# agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")


from smolagents import CodeAgent
from smolagents.models import TransformersModel

# model = VLLMModel(
#     model_id="mistralai/Mistral-7B-Instruct-v0.3",
#     model_kwargs={"revision": "main", "gpu_memory_utilization":0.6},
# )

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

model = TransformersModel(model_id, device_map='cuda:2')
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, max_steps=3)
result = agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
print(result)