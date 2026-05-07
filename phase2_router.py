from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore") 

# 1. State
class State(TypedDict):
    bot_id: str
    topic: str
    search_results: str
    post_content: str

# 2. LLM
load_dotenv()  # Load environment variables from .env file

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile" #changing the model as llama3-70b-8192 is not available in the Groq API anymore. This is the closest one available. 
)

# 3. Mock Search Tool
def mock_search(query: str) -> str:
    if "crypto" in query.lower():
        return "Bitcoin hits all time high"
    elif "ai" in query.lower():
        return "OpenAI releases new model"
    else:
        return "Markets are volatile today"

# 4. Node 1
def decide_topic(state: State) -> dict:
    persona = state["bot_id"]      # Store the reply in state["topic"]
    messages = [
        SystemMessage(content=f"You are {persona}"),
        HumanMessage(content="Decide one topic according to your persona. One sentence only. No extra explanation.")
    ]                              # fill the prompt with SystemMessage and HumanMessage as needed

    response = llm.invoke(messages)   
    return {"topic": response.content}

# 5. Node 2
def search_news(state: State) -> dict:
    topic = state["topic"]        # Use the topic from state to search news
    results = mock_search(topic)  # Store the search results in state["search_results"]
    return {"search_results": results}

# 6. Node 3
def write_post(state: State) -> dict:
    persona = state["bot_id"]
    topic = state["topic"]
    search_results = state["search_results"]  # Use the search results to write a post
    messages = [
        SystemMessage(content=f"You are {persona}"),
        HumanMessage(content=f"Write a social media post about {topic} based on this news: {search_results}. Write a 280-char post.")
    ]
    response = llm.invoke(messages)
    return {"post_content": response.content}

# 7. Build Graph
graph = StateGraph(State)
graph.add_node("decide_topic", decide_topic)
graph.add_node("search_news", search_news)
graph.add_node("write_post", write_post)

graph.add_edge(START, "decide_topic")
graph.add_edge("decide_topic", "search_news")
graph.add_edge("search_news", "write_post")
graph.add_edge("write_post", END)

app = graph.compile()

if __name__ == "__main__":
    bots = [
        "Bot A (Tech Maximalist): I believe AI and crypto will solve all human problems.",
        "Bot B (Doomer/Skeptic): I believe tech monopolies are destroying society.",
        "Bot C (Finance Bro): I strictly care about markets and making money."
    ]

    for bot in bots:
        result = app.invoke({
            "bot_id": bot,
            "topic": "",
            "search_results": "",
            "post_content": ""
        })
        print(f"\n--- {bot[:5]} ---")
        print("Post:", result["post_content"])