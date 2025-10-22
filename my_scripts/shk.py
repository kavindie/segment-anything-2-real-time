from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

model = ChatOllama(
    ## try `$ ollama list` to see whats downloaded
    ## or <https://ollama.com/search?c=thinking&c=tools> to find more
    # model = "llama3",  # didn't try actually
    # model = "mistral",  # shitty (but uses tools)
    # model = "deepseek-r1", # mixed reports of tool usage, probably possible to /hack/
    # model = "magistral",  # got it right but didn't use tools!
    model = "qwen3",  # got it very almost right (copy error), but did most of the calcs in it's head!
    # other params ...
    # temperature = 0.8,
    # num_predict = 256,
)

## TEST ChatOllama
# messages = [
#     ("system", "You are a helpful translator. Translate the user sentence to French."),
#     ("human", "I love programming."),
# ]
# print(model.invoke(messages).content)

tools = []

@tools.append
@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

@tools.append
@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int

@tools.append
@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent

## Create a React agent with the model and tools
agent = create_react_agent(
    model=model,
    tools=tools,
)

query = (
    "Take 3 to the fifth power and multiply that by the sum of twelve and "
    "three, then square the whole result."
)
input_message = {"role": "user", "content": query}

for step in agent.stream({"messages": [input_message]}, stream_mode="values"):
    step["messages"][-1].pretty_print()

## In [1]: (3**5 * (12 + 3))**2
## Out[1]: 13286025
