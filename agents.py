from typing import Annotated, Dict, TypedDict, List
from langgraph.graph import Graph
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
import json
import re

class AgentState(TypedDict):
    messages: List[BaseMessage]
    article: str
    topic: str
    openai_analysis: Dict
    anthropic_analysis: Dict
    final_comparison: Dict

def extract_json_from_response(response_text: str) -> Dict:
    """Extract JSON from response that might be wrapped in markdown code blocks."""
    # Try to extract JSON from code blocks first
    json_match = re.search(r"```(?:json)?\n(.*?)\n```", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # If no code block or parsing failed, try parsing the whole response
    try:
        return json.loads(response_text)
    except:
        return {
            "error": "Failed to parse response",
            "raw_response": response_text
        }

def create_analysis_prompt(article: str, topic: str = None) -> str:
    topic_context = topic if topic else "the main theme suggested by this content"
    
    return f"""Analyze the following article and provide detailed, actionable feedback:

Article:
{article}

Target Topic/Search Term: {topic_context}

Please provide a structured analysis covering:

1. contentQuality:
   - depth: 
     * rating: (Excellent/Good/Fair/Poor)
     * analysis: How well does it explore the main ideas? What key aspects are missing?
     * missingElements: List specific missing elements
   - accuracy:
     * rating: (Excellent/Good/Fair/Poor) 
     * logicalIssues: Are the arguments logically sound? Any unsupported claims?
   - originality:
     * rating: (Excellent/Good/Fair/Poor)
     * uniqueElements: What makes this perspective unique or valuable?
     * limitations: What prevents it from being more original?

2. writingStyle:
   - clarity:
     * rating: (Excellent/Good/Fair/Poor)
     * issues: Are ideas presented logically? Any confusing sections?
   - engagement:
     * rating: (Excellent/Good/Fair/Poor)
     * weaknesses: What specific elements make it compelling or where does it fall flat?
   - flow:
     * rating: (Excellent/Good/Fair/Poor)
     * issues: How well do paragraphs connect? Where are transitions weak?

3. contentEnhancement:
   - keyTermsNeeded: What important concepts should be better explained?
   - examplesNeeded: Where could specific examples strengthen arguments?
   - counterArgumentsNeeded: What opposing viewpoints should be addressed?
   - supportingEvidenceNeeded: What data or research could reinforce points?

4. searchRankingAnalysis:
   - relevanceScore: (0-10) How well does this match the target topic?
   - currentRankingPrediction: Where would this rank on page 1-10+ for the topic?
   - competitorAnalysis: What type of content currently ranks well for this topic?
   - rankingImprovement: Specific changes needed to rank higher
   - searchIntent: Does this match what people searching for this topic want?
   - keywordGaps: What important keywords/phrases are missing?

5. improvementActions:
   - Each item should have: priority (High/Medium/Low), action, impact
   - Provide at least 5 concrete, actionable steps
   - Focus on changes that would improve search ranking AND content quality

6. qualityAssessment:
   - score: (0-100)
   - explanation: Brief explanation of the score
   - strengths: Top 3 strengths  
   - immediateImprovements: Top 3 areas needing immediate improvement

Format your response as a JSON object with these exact keys. Do not wrap the JSON in code blocks. Be specific and actionable in your suggestions, especially for search ranking improvements."""

def analyze_with_openai(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(create_analysis_prompt(state["article"], state.get("topic")))
    response = llm.invoke(prompt.format_messages())
    state["openai_analysis"] = extract_json_from_response(response.content)
    return state

def analyze_with_anthropic(state: AgentState) -> AgentState:
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)
    prompt = ChatPromptTemplate.from_template(create_analysis_prompt(state["article"], state.get("topic")))
    response = llm.invoke(prompt.format_messages())
    state["anthropic_analysis"] = extract_json_from_response(response.content)
    return state

def compare_and_summarize(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    comparison_prompt = ChatPromptTemplate.from_template(
        """Compare the following two analyses of an article and provide a comprehensive improvement plan:

OpenAI Analysis:
{openai_analysis}

Anthropic Analysis:
{anthropic_analysis}

Target Topic: {topic}

Please provide:

1. keyInsights:
   - agreement: Where do the analyses agree?
   - disagreement: Where do they differ significantly?
   - mostActionableFeedback: Which model provided more specific, actionable advice?

2. searchRankingStrategy:
   - currentPosition: Based on both analyses, where would this content likely rank?
   - topCompetitors: What type of content would outrank this?
   - quickWins: 3 fastest changes to improve ranking
   - longTermStrategy: Major content additions needed for top 3 ranking
   - contentGaps: What's missing compared to top-ranking content?

3. prioritizedImprovementPlan:
   - immediateActions: (High impact, both models agree)
     * List specific changes with exact implementation steps
   - secondaryImprovements: (High value from either analysis)
     * Include specific examples and recommendations  
   - optionalEnhancements: (Lower priority but still valuable)

4. contentPositioning:
   - bestAspectsToPreserve: What should definitely be kept?
   - criticalAreasToRevise: What must be changed for better ranking?
   - uniqueAngle: How to differentiate from competitors?
   - targetAudience: Who should this content serve?

5. finalAssessment:
   - combinedQualityScore: (Weighted average with explanation)
   - combinedRankingPrediction: Realistic ranking expectation after improvements
   - confidenceLevel: How confident are you in these recommendations?
   - expectedTimeToRank: How long might improvements take to show results?

Format your response as a JSON object. Do not wrap the JSON in code blocks. Focus on specific, actionable improvements that will help with both content quality and search ranking."""
    )
    
    response = llm.invoke(
        comparison_prompt.format_messages(
            openai_analysis=json.dumps(state["openai_analysis"], indent=2),
            anthropic_analysis=json.dumps(state["anthropic_analysis"], indent=2),
            topic=state.get("topic", "the main theme")
        )
    )
    
    state["final_comparison"] = extract_json_from_response(response.content)
    return state

def create_analysis_graph() -> Graph:
    workflow = Graph()
    
    # Define the nodes
    workflow.add_node("openai_analysis", analyze_with_openai)
    workflow.add_node("anthropic_analysis", analyze_with_anthropic)
    workflow.add_node("compare", compare_and_summarize)
    
    # Define the edges
    workflow.add_edge("openai_analysis", "anthropic_analysis")
    workflow.add_edge("anthropic_analysis", "compare")
    
    # Set the entry point
    workflow.set_entry_point("openai_analysis")
    
    # Set the exit point
    workflow.set_finish_point("compare")
    
    # Compile the graph into a runnable chain
    return workflow.compile() 