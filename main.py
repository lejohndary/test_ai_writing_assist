from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from agents import create_analysis_graph, AgentState
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Check for required API keys
if not os.getenv("OPENAI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
    raise EnvironmentError("Missing required API keys. Please set OPENAI_API_KEY and ANTHROPIC_API_KEY in .env file")

app = FastAPI(
    title="Article Analysis Agent",
    description="API for analyzing articles using LangGraph with OpenAI and Anthropic models",
    version="1.0.0"
)

class ArticleRequest(BaseModel):
    text: str
    topic: str = None  # Optional topic

class AnalysisResponse(BaseModel):
    openai_analysis: Dict
    anthropic_analysis: Dict
    final_comparison: Dict

def analyze_text(text: str, topic: str = None) -> Dict:
    """Analyze text directly without going through the API"""
    if len(text.strip()) < 50:
        raise ValueError("Article text is too short. Please provide at least 50 characters.")
    
    chain = create_analysis_graph()
    initial_state = AgentState(
        messages=[],
        article=text,
        topic=topic,
        openai_analysis={},
        anthropic_analysis={},
        final_comparison={}
    )
    
    final_state = chain.invoke(initial_state)
    return {
        "openai_analysis": final_state["openai_analysis"],
        "anthropic_analysis": final_state["anthropic_analysis"],
        "final_comparison": final_state["final_comparison"]
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_article(request: ArticleRequest):
    try:
        return analyze_text(request.text, request.topic)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Article Analysis Agent API",
        "usage": "Send a POST request to /analyze with your article text"
    }

if __name__ == "__main__":
    # Example usage without API
    sample_text = """
Technology does many things, but one thing it never does is save time

It either reduces headcount, while amplifying the productivity of the remaining workers, so that fewer people working the same amount as before output the same total amount of products as before in the same amount of time

Or it keeps headcount, so that the same number of people output more products in the same amount of time

Whether its the former or the latter depends on how much potential for growth there is in the economy. If you have hit a growth wall, tech will just create unemployment. If you have not, it might unlock new production

In either case, what it will never do is keep the same number of people while allowing them to output the same amount of products while doing less work.

At a systemic level, a capitalist system will always convert tech into either new output with no increased leisure, or the same output with reduced headcount. Those are your options, so the next time you see one of those god-damn Silicon Valley futurists spinning illusory tales of AI liberating us from work, you need to see it for what it is: propaganda
"""
    
    # Specify a topic for better analysis
    topic = "technology productivity and time savings"
    
    try:
        result = analyze_text(sample_text, topic)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}") 