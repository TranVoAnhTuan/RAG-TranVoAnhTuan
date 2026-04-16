from .state import IngestionState
from langchain_ollama import ChatOllama
from app.core.config import settings

async def tag_node(state: IngestionState):
    print("🏷️ Auto-tagging document...")
    raw_text = state.get("raw_text", "")
    
    first_page_text = raw_text[:2000]
    
    topics_dict = settings.AVAILABLE_TOPICS
    topic_names = list(topics_dict.keys())

    topic_descriptions_str = "\n".join([f"- **{k}**: {v}" for k, v in topics_dict.items()])
    
    prompt = f"""
    Read the following start of a document and classify it into EXACTLY ONE of these topics.
    
    Here are the available topics and their descriptions:
    {topic_descriptions_str}
    
    Return ONLY the exact topic name from this list: {topic_names}. 
    Do not add quotes, formatting, or explanations.
    
    DOCUMENT TEXT:
    {first_page_text}
    """
    
    try:
        llm = ChatOllama(
            model=settings.LLM_MODEL, 
            temperature=0,
            base_url=settings.BASE_URL
        )
        response = await llm.ainvoke(prompt)
        assigned_topic = response.content.strip()
        
        if assigned_topic not in topic_names:
            assigned_topic = "General" if "General" in topic_names else topic_names[-1]
            
    except Exception as e:
        print(f"Tagging error: {e}")
        assigned_topic = "General" if "General" in topic_names else topic_names[-1]        
    print(f"✅ Document auto-tagged as: {assigned_topic}")
    return {"topic": assigned_topic}