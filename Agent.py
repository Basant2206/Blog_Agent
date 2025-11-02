import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# ============ LOAD ENV ============
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============ DEFINE STATE ============
class BlogState(TypedDict):
    topic: str
    research: str
    outline: str
    content: str
    final_blog: str

# ============ LLM SETUP ============
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  
    temperature=0.7,
)

# ============ TOOLS ============
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000))
duckduck_tool = DuckDuckGoSearchResults()

# ============ NODES ============

def research_node(state: BlogState):
    """Gather research from Wikipedia and DuckDuckGo"""
    topic = state["topic"]
    print(f"\n Researching topic: {topic}")

    wiki_data = wikipedia_tool.run(topic)
    web_data = duckduck_tool.run(topic)

    research_text = f"Wikipedia info:\n{wiki_data}\n\nDuckDuckGo info:\n{web_data}"
    return {"research": research_text}

def outline_node(state: BlogState):
    """Create an outline for the blog"""
    research = state["research"]
    topic = state["topic"]
    print("\n Generating outline...")

    prompt = [
        SystemMessage(content="You are a blog planning assistant."),
        HumanMessage(
            content=f"Based on the following research, create a detailed outline for a blog about '{topic}':\n\n{research}"
        ),
    ]
    outline = llm.invoke(prompt).content
    return {"outline": outline}

def content_node(state: BlogState):
    """Generate full blog content based on outline"""
    topic = state["topic"]
    outline = state["outline"]
    print("\n Writing blog content...")

    prompt = [
        SystemMessage(content="You are a professional blog writer."),
        HumanMessage(
            content=f"Write a detailed blog on '{topic}' following this outline:\n{outline}\n"
                    "Include the following sections:\n"
                    "1. Heading\n2. Introduction\n3. Content\n4. Summary"
        ),
    ]
    content = llm.invoke(prompt).content
    return {"content": content}

def finalize_node(state: BlogState):
    """Combine everything into a structured blog output"""
    print("\n Finalizing blog...")
    return {"final_blog": state["content"]}

# ============ BUILD GRAPH ============
graph = StateGraph(BlogState)

graph.add_node("research", research_node)
graph.add_node("outline", outline_node)
graph.add_node("content", content_node)
graph.add_node("finalize", finalize_node)

# ============ EDGES ============
graph.set_entry_point("research")
graph.add_edge("research", "outline")
graph.add_edge("outline", "content")
graph.add_edge("content", "finalize")
graph.add_edge("finalize", END)

# ============ COMPILE GRAPH ============
app = graph.compile()

# ============ RUN FUNCTION ============
def generate_blog(topic: str):
    print("\n Starting Blog Generation System...")
    inputs = {"topic": topic}
    final_state = app.invoke(inputs)
    print("\n Blog generation complete!\n")
    print("=" * 60)
    print(final_state["final_blog"])
    print("=" * 60)
    return final_state["final_blog"]

# ============ MAIN ============
if __name__ == "__main__":
    topic = input("Enter a blog topic: ")
    generate_blog(topic)
