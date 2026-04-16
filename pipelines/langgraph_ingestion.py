from langgraph.graph import StateGraph, START, END
from .state import IngestionState
from .extract import extract_node
from .tag import tag_node
from .clean import clean_node
from .chunk import chunk_node
from .embed import embed_and_load_node
from .check_duplicate import check_duplicate_node

def route_after_check(state: IngestionState):
    """Conditional routing based on document duplication"""
    if state.get("is_duplicate", False):
        return END
    return "extract"

def build_ingestion_graph():
    workflow = StateGraph(IngestionState)

    workflow.add_node("check_duplicate", check_duplicate_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("tag", tag_node) 
    workflow.add_node("clean", clean_node)
    workflow.add_node("chunk", chunk_node)
    workflow.add_node("embed_and_load", embed_and_load_node)

    workflow.add_edge(START, "check_duplicate")
    
    workflow.add_conditional_edges(
        "check_duplicate",
        route_after_check,
        {END: END, "extract": "extract"}
    )

    workflow.add_edge("extract", "tag")
    workflow.add_edge("tag", "clean")
    workflow.add_edge("clean", "chunk")
    workflow.add_edge("chunk", "embed_and_load")
    workflow.add_edge("embed_and_load", END)

    return workflow.compile()

ingestion_app = build_ingestion_graph()