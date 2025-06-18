from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_huggingface import HuggingFaceEndpoint

import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

import os
from huggingface_hub import InferenceClient
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Method 1: Direct wrapper to match your existing code structure
class HuggingFaceChatWrapper:
    def __init__(self, model_name="sarvamai/sarvam-m"):
        self.client = InferenceClient(
            provider="hf-inference",
        )
        self.model_name = model_name
    
    def invoke(self, messages):
        """Convert LangChain messages to HF chat format and get response"""
        
        # Convert LangChain messages to HF format
        if isinstance(messages, str):
            # If it's just a string, treat as user message
            hf_messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            hf_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    hf_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    hf_messages.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, SystemMessage):
                    hf_messages.append({"role": "system", "content": msg.content})
                else:
                    # Fallback for other message types
                    hf_messages.append({"role": "user", "content": str(msg.content)})
        else:
            # Single message object
            if hasattr(messages, 'content'):
                hf_messages = [{"role": "user", "content": messages.content}]
            else:
                hf_messages = [{"role": "user", "content": str(messages)}]
        
        # Make the API call
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=hf_messages,
        )
        
        # Return the response content
        return completion.choices[0].message.content

# Method 2: Your exact code adapted for LangChain messages
def hf_chat_invoke(messages, model_name="sarvamai/sarvam-m"):
    """Function that works exactly like your code but accepts LangChain messages"""
    
    client = InferenceClient(
        provider="hf-inference",
    )
    
    # Convert LangChain message to HF format
    if isinstance(messages, list) and len(messages) > 0:
        content = messages[0].content
    elif hasattr(messages, 'content'):
        content = messages.content
    else:
        content = str(messages)
    
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
    )
    
    return completion.choices[0].message.content

# LLM model
model = HuggingFaceChatWrapper("sarvamai/sarvam-m")


class EmailState(TypedDict):
    # The email being processed
    email: Dict[str, Any]  # Contains subject, sender, body, etc.

    # Category of the email (inquiry, complaint, etc.)
    email_category: Optional[str]

    # Reason why the email was marked as spam
    spam_reason: Optional[str]

    # Analysis and decisions
    is_spam: Optional[bool]
    
    # Response generation
    email_draft: Optional[str]
    
    # Processing metadata
    messages: List[Dict[str, Any]]  # Track conversation with LLM for analysis


def read_email(state: EmailState):
    """Alfred reads and logs the incoming email"""
    email = state["email"]
    
    # Here we might do some initial preprocessing
    print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")
    
    # No state changes needed here
    return {}

def classify_email(state: EmailState):
    """Alfred uses an LLM to determine if the email is spam or legitimate"""
    email = state["email"]
    
    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, analyze this email and determine if it is spam or legitimate.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    First, determine if this email is spam. If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).
    """
    
    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = hf_chat_invoke(messages, "sarvamai/sarvam-m") #model.invoke(messages)
    print(response)
    
    # Simple logic to parse the response (in a real app, you'd want more robust parsing)
    response_text = response.lower()
    is_spam = "spam" in response_text and "not spam" not in response_text
    
    # Extract a reason if it's spam
    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()
    
    # Determine category if legitimate
    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break
    
    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    
    # Return state updates
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }

def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    print(f"Alfred has marked the email as spam. Reason: {state['spam_reason']}")
    print("The email has been moved to the spam folder.")
    
    # We're done processing this email
    return {}

def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    category = state["email_category"] or "general"
    
    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    This email has been categorized as: {category}
    
    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """
    
    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = hf_chat_invoke(messages, "sarvamai/sarvam-m") #model.invoke(messages)
    
    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    
    # Return state updates
    return {
        "email_draft": response,
        "messages": new_messages
    }

def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response"""
    email = state["email"]
    
    print("\n" + "="*50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-"*50)
    print(state["email_draft"])
    print("="*50 + "\n")
    
    # We're done processing this email
    return {}

def route_email(state: EmailState) -> str:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"

# Create the graph
email_graph = StateGraph(EmailState)

# Add nodes
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_response", draft_response)
email_graph.add_node("notify_mr_hugg", notify_mr_hugg)

# Start the edges
email_graph.add_edge(START, "read_email")
# Add edges - defining the flow
email_graph.add_edge("read_email", "classify_email")

# Add conditional branching from classify_email
email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {
        "spam": "handle_spam",
        "legitimate": "draft_response"
    }
)

# Add the final edges
email_graph.add_edge("handle_spam", END)
email_graph.add_edge("draft_response", "notify_mr_hugg")
email_graph.add_edge("notify_mr_hugg", END)

# Compile the graph
compiled_graph = email_graph.compile()

display(Image(compiled_graph.get_graph().draw_mermaid_png()))

# Example legitimate email
legitimate_email = {
    "sender": "john.smith@monash.edu",
    "subject": "Question about your services",
    "body": "Dear Mr. Hugg, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith"
}

legitimate_result = compiled_graph.invoke({
    "email": legitimate_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})


print(legitimate_result)