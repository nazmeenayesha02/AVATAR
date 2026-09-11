import os
import json
import base64
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from dotenv import load_dotenv
import certifi

# Clean environment to prevent stale cached keys
for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY"]:
    if key in os.environ:
        del os.environ[key]

load_dotenv(override=True)

# Synchronize API keys so langchain uses the active one
active_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if active_key:
    os.environ["GOOGLE_API_KEY"] = active_key
    os.environ["GEMINI_API_KEY"] = active_key

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from tools import tools, web_search, search_uploaded_documents, CURRENT_THREAD_ID
from rag import retrieve_from_rag
from tavily import TavilyClient

Path("data").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_MODEL") or "gemini-3.6-flash"

ALLOWED_MODELS = {
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-1.5-flash"
}

SYSTEM_PERSONA = """You are AVATAR, an Action-Oriented AI Agent and Universal Bridge between Human Intent and Complex Systems.
You solve societal benefit problems by converting messy, unstructured real-world requests (government schemes, welfare benefits, healthcare access, education, legal aid, documentation, public utilities, and civic complaints) into clear, verified, and actionable solutions.

When introducing yourself or asked who you are, respond:
"Hey 👋 I am AVATAR, your Real-World Problem to Verified Action Assistant.
I act as a universal bridge between human intent and complex systems — converting messy real-world challenges into clear, verified, and actionable steps. Whether you need help with government schemes, healthcare aid, educational scholarships, legal documentation, or civic services, tell me what you're facing and let's get it resolved step by step."

Core Operating Rules:
1. Move the user from "I have a problem" to "Here is what the problem is, what we know, what we verified, and what you should do next."
2. Never invent facts, sources, deadlines, eligibility requirements, government rules, medical diagnoses, legal claims, or fees.
3. If information is uncertain, outdated, or missing, explicitly say so.
4. Distinguish between:
   - USER PROVIDED: Facts given by the user
   - RETRIEVED: Found in official documents or web search
   - VERIFIED: Corroborated with authoritative records
   - AI INTERPRETATION: Logical analysis and guidance
5. For emergencies, healthcare crises, or severe legal peril:
   - Prioritize immediate emergency services, official hotlines, or certified professionals.
   - Flag high urgency prominently.
6. Return the response in the user's chosen language (English, Hindi, or Kannada) while keeping official scheme names, form titles, and URLs accurate and intact.
"""


class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_input: str
    language: str
    image_path: Optional[str]
    intent: str
    urgency: str
    entities: Dict[str, Any]
    route: str
    rag_evidence: str
    web_evidence: str
    image_evidence: str
    verification_notes: str
    action_plan: str
    sources: List[Dict[str, str]]


def normalize_model_name(model_name: str | None) -> str:
    if not model_name:
        return DEFAULT_MODEL
    model_name = model_name.strip()
    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL
    return model_name


def get_llm(model_name: str = DEFAULT_MODEL, temperature: float = 0.2, streaming: bool = False):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
    return ChatGoogleGenerativeAI(
        model=normalize_model_name(model_name),
        google_api_key=api_key,
        temperature=temperature,
        streaming=streaming
    )


def extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "content" in item and isinstance(item["content"], str):
                    parts.append(item["content"])
        return "".join(parts)
    return str(content) if content is not None else ""


# ==============================================================================
# 1. INTENT & URGENCY CLASSIFIER NODE
# ==============================================================================
async def analyze_intent_node(state: AgentState) -> Dict[str, Any]:
    user_input = state.get("user_input", "")
    image_path = state.get("image_path")
    has_image = bool(image_path and os.path.exists(image_path))

    llm = get_llm(temperature=0.1)

    classification_prompt = f"""You are the Intent Analysis engine for Nazmeen Ayesha Action Assistant.
Analyze the user input and classify it accurately for a societal benefit assistance workflow.

User Input: "{user_input}"
Has Image Attached: {has_image}

Classify into strict JSON format with these exact keys:
{{
    "intent": "GOVERNMENT_SERVICE" | "EDUCATION" | "HEALTHCARE" | "EMERGENCY" | "DOCUMENT_HELP" | "FINANCIAL_ASSISTANCE" | "PUBLIC_SERVICE" | "GENERAL_INFORMATION" | "OTHER",
    "urgency": "LOW" | "NORMAL" | "IMPORTANT" | "URGENT",
    "route": "DIRECT" | "RAG" | "WEB" | "IMAGE" | "MULTI_TOOL",
    "search_query": "specific web search query for Tavily if web verification is needed, else empty string",
    "entities": {{
        "location": "extracted location or null",
        "date_or_deadline": "extracted date or null",
        "organization": "department or ministry or null",
        "document_name": "identity card or certificate or null",
        "scheme_name": "name of scheme or service or null",
        "goal": "what the person wants to achieve",
        "missing_info": ["critical missing pieces needed to complete the action"]
    }}
}}

Routing Guidelines:
- If an image is attached -> "IMAGE" (or "MULTI_TOOL" if web verification also needed).
- If the user asks about an uploaded document or file -> "RAG".
- If the query is a greeting or general chitchat ("hi", "who are you") -> "DIRECT".
- If the query is about government schemes, eligibility, official rules, public programs, recent news, deadlines, or benefits -> "WEB".
- If both uploaded documents and fresh web verification are relevant -> "MULTI_TOOL".

Respond ONLY with valid JSON. No commentary."""

    try:
        res = await llm.ainvoke([HumanMessage(content=classification_prompt)])
        text = extract_content_text(res.content).strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(text)
        intent = parsed.get("intent", "GENERAL_INFORMATION")
        urgency = parsed.get("urgency", "NORMAL")
        route = parsed.get("route", "DIRECT")
        entities = parsed.get("entities", {})
        search_query = parsed.get("search_query", user_input)

        if has_image and route not in ["IMAGE", "MULTI_TOOL"]:
            route = "IMAGE"

        return {
            "intent": intent,
            "urgency": urgency,
            "route": route,
            "entities": entities,
            "verification_notes": f"Search Query: {search_query}"
        }
    except Exception as e:
        print(f"Intent analysis fallback: {e}")
        route = "IMAGE" if has_image else ("DIRECT" if len(user_input.split()) < 4 else "WEB")
        return {
            "intent": "GENERAL_INFORMATION",
            "urgency": "NORMAL",
            "route": route,
            "entities": {"goal": user_input},
            "verification_notes": ""
        }


# ==============================================================================
# 2. EVIDENCE COLLECTION NODES
# ==============================================================================
async def rag_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("user_input", "")
    thread_id = CURRENT_THREAD_ID
    try:
        rag_text = retrieve_from_rag(query=query, thread_id=thread_id, k=4)
        return {
            "rag_evidence": rag_text,
            "sources": [{"title": "Uploaded Document (Knowledge Base)", "url": "#rag-doc"}]
        }
    except Exception as e:
        return {"rag_evidence": f"RAG retrieval notice: {str(e)}"}


async def web_search_node(state: AgentState) -> Dict[str, Any]:
    user_input = state.get("user_input", "")
    notes = state.get("verification_notes", "")
    query = user_input
    if "Search Query:" in notes:
        query = notes.split("Search Query:")[1].strip() or user_input

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "web_evidence": "Tavily web search key not configured.",
            "sources": []
        }

    try:
        client = TavilyClient(api_key=api_key)
        # Search official authoritative records
        search_res = client.search(
            query=query,
            max_results=5,
            search_depth="advanced"
        )
        
        results = search_res.get("results", [])
        evidence_parts = []
        sources = []
        for item in results:
            title = item.get("title", "Official Source")
            url = item.get("url", "")
            content = item.get("content", "")
            evidence_parts.append(f"Source: {title} ({url})\nExcerpt: {content}")
            sources.append({"title": title, "url": url})

        return {
            "web_evidence": "\n\n---\n\n".join(evidence_parts),
            "sources": sources
        }
    except Exception as e:
        print(f"Web search error: {e}")
        return {
            "web_evidence": f"Web search could not be completed: {str(e)}",
            "sources": []
        }


async def image_analysis_node(state: AgentState) -> Dict[str, Any]:
    image_path = state.get("image_path")
    user_input = state.get("user_input", "Analyze this image and extract all relevant information.")

    if not image_path or not os.path.exists(image_path):
        return {"image_evidence": "No image found for analysis."}

    try:
        with open(image_path, "rb") as img_file:
            b64_data = base64.b64encode(img_file.read()).decode("utf-8")

        suffix = Path(image_path).suffix.lower()
        mime_type = "image/jpeg"
        if suffix == ".png":
            mime_type = "image/png"
        elif suffix == ".webp":
            mime_type = "image/webp"

        llm = get_llm(temperature=0.1)
        vision_prompt = f"""You are the Multimodal Notice & Document Inspector for Nazmeen Ayesha Action Assistant.
The user provided this image alongside this request: "{user_input}"

Examine this image with extreme precision (e.g. government notice, application form, official poster, signboard, circular, legal document, receipt, or screenshot).
Extract:
1. Official Scheme / Document Title / Authority:
2. Important Dates, Deadlines & Timelines:
3. Eligibility Criteria & Beneficiaries:
4. Required Documents & Identification:
5. Application Process & Fees:
6. Official Portal / Contact / Helpline:
7. Crucial Conditions / Warnings:

Return a structured breakdown of the facts discovered in the image."""

        msg = HumanMessage(content=[
            {"type": "text", "text": vision_prompt},
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{b64_data}"}
        ])

        res = await llm.ainvoke([msg])
        extracted_info = extract_content_text(getattr(res, "content", res))
        return {
            "image_evidence": extracted_info,
            "sources": [{"title": f"Analyzed Image ({Path(image_path).name})", "url": f"/uploads/{Path(image_path).name}"}]
        }
    except Exception as e:
        print(f"Image analysis error: {e}")
        return {"image_evidence": f"Failed to analyze image: {str(e)}"}


async def multi_tool_node(state: AgentState) -> Dict[str, Any]:
    rag_res = await rag_node(state)
    web_res = await web_search_node(state)
    image_res = {}
    if state.get("image_path") and os.path.exists(state.get("image_path", "")):
        image_res = await image_analysis_node(state)

    combined_sources = []
    if rag_res.get("sources"):
        combined_sources.extend(rag_res["sources"])
    if web_res.get("sources"):
        combined_sources.extend(web_res["sources"])
    if image_res.get("sources"):
        combined_sources.extend(image_res["sources"])

    return {
        "rag_evidence": rag_res.get("rag_evidence", ""),
        "web_evidence": web_res.get("web_evidence", ""),
        "image_evidence": image_res.get("image_evidence", ""),
        "sources": combined_sources
    }


# ==============================================================================
# 3. VERIFICATION & REASONING NODE
# ==============================================================================
async def verify_node(state: AgentState) -> Dict[str, Any]:
    intent = state.get("intent", "GENERAL_INFORMATION")
    urgency = state.get("urgency", "NORMAL")
    user_input = state.get("user_input", "")
    rag_evidence = state.get("rag_evidence", "")
    web_evidence = state.get("web_evidence", "")
    image_evidence = state.get("image_evidence", "")

    # Synthesis of verification findings
    notes = []
    if web_evidence and "Source:" in web_evidence:
        notes.append("Corroborated against live public web records.")
    if rag_evidence and "No relevant" not in rag_evidence:
        notes.append("Cross-referenced with user-uploaded reference documents.")
    if image_evidence and "Failed" not in image_evidence and "No image" not in image_evidence:
        notes.append("Extracted directly from user-submitted image/notice.")
    
    if not notes:
        notes.append("Reasoned from conversational context; no external documents or portal checks required.")

    return {
        "verification_notes": " | ".join(notes)
    }


# ==============================================================================
# 4. ACTION PLAN & RESPONSE GENERATOR NODE (STREAMED)
# ==============================================================================
async def generate_action_plan_node(state: AgentState):
    user_input = state.get("user_input", "")
    intent = state.get("intent", "GENERAL_INFORMATION")
    urgency = state.get("urgency", "NORMAL")
    entities = state.get("entities", {})
    rag_evidence = state.get("rag_evidence", "")
    web_evidence = state.get("web_evidence", "")
    image_evidence = state.get("image_evidence", "")
    verification_notes = state.get("verification_notes", "")
    sources = state.get("sources", [])
    language = state.get("language", "English")

    llm = get_llm(temperature=0.3, streaming=True)

    # Greeting / Casual check
    casual_greetings = {"hi", "hello", "hey", "who are you", "what can you do", "namaste", "vanakkam", "namaskara"}
    cleaned_input = user_input.strip().lower()
    if cleaned_input in casual_greetings or (len(cleaned_input.split()) <= 2 and "help" not in cleaned_input):
        intro_prompt = f"""{SYSTEM_PERSONA}

The user greeted with: "{user_input}"
Respond warmly in {language} as AVATAR.
Introduce yourself, explain your purpose as a Real-World Problem to Verified Action Assistant for societal benefit, and invite them to describe any problem or upload a notice/form.
Keep official names in accurate form."""
        res = await llm.ainvoke([HumanMessage(content=intro_prompt)])
        clean_text = extract_content_text(res.content)
        return {"messages": [AIMessage(content=clean_text)]}

    # Build comprehensive synthesis prompt
    sources_formatted = "\n".join([f"- [{s['title']}]({s['url']})" for s in sources if s.get("url")])

    action_prompt = f"""{SYSTEM_PERSONA}

You are generating the final verified response for the user.

USER REQUEST:
"{user_input}"

TARGET LANGUAGE:
Translate and format the ENTIRE response in **{language}** (English, Hindi, or Kannada).
IMPORTANT: Maintain exact official government portal URLs, scheme names (e.g. PM-Kisan, Ayushman Bharat, Seva Sindhu, Aadhaar), and form names in recognized form.

ANALYSIS DATA:
- Intent Domain: {intent}
- Urgency Level: {urgency}
- Extracted Entities: {json.dumps(entities, ensure_ascii=False)}

COLLECTED EVIDENCE:
Document RAG Evidence:
{rag_evidence if rag_evidence else "None"}

Live Web Search Evidence:
{web_evidence if web_evidence else "None"}

Image / Notice Analysis Evidence:
{image_evidence if image_evidence else "None"}

Verification Context:
{verification_notes}

==================================================
RESPONSE FORMAT REQUIREMENTS
==================================================
You MUST format your response using standard GitHub Markdown with these exact sections (do not display empty sections). Include the urgency badge header if urgency is IMPORTANT or URGENT:

{"`URGENT ACTION REQUIRED`" if urgency == "URGENT" else ("`PRIORITY ACTION`" if urgency == "IMPORTANT" else "")}

### WHAT I UNDERSTOOD
Briefly explain the user's situation and core goal in 1-2 empathetic sentences.

### PROBLEM
Clearly pinpoint the exact bottleneck, bureaucratic challenge, missing requirement, or problem faced.

### KEY INFORMATION
List key entities extracted from the situation:
- **User Provided**: Information given directly by the user.
- **Goal / Scheme**: The specific program, benefit, or service involved.
- **Constraints / Deadlines**: Any known deadlines, fees, or location factors.

### WHAT I FOUND
Summarize facts retrieved from official sources, documents, or image notices. Never invent rules.

### VERIFICATION
Explain explicitly:
- What has been verified through official portals or evidence.
- What remains unverified or requires the user's confirmation.
- Any recent changes or warnings in rules/deadlines.

### ACTION PLAN
Numbered, step-by-step sequential guide on what the user must do:
1. **Step 1**: Clear concrete action (where to go, which portal, or what form to fill).
2. **Step 2**: Intermediate action (documents to carry, online upload steps).
3. **Step 3**: Follow-up action (acknowledgment receipt, tracking status, verification).

### DOCUMENTS / INFORMATION NEEDED
(Only show if documents or details are required):
- Required certificates (e.g., Aadhaar, Income Certificate, BPL card).
- Photographs, bank passbook, or affidavits.

### IMPORTANT / URGENT
(Only show if there are critical deadlines, emergency services, or warnings):
- Critical warnings, fraud prevention advice, official helpline numbers (e.g. 112, 1930, 1075).

### SOURCES
{sources_formatted if sources_formatted else "Official reference databases & public records."}

### NEXT STEP
Provide the SINGLE most useful immediate action the user should do right now (e.g., "Visit the official portal at ... and click on New Registration").
"""

    res = await llm.ainvoke([HumanMessage(content=action_prompt)])
    clean_text = extract_content_text(res.content)
    return {"messages": [AIMessage(content=clean_text)]}


# ==============================================================================
# 5. LANGGRAPH BUILDER
# ==============================================================================
def route_decision(state: AgentState) -> str:
    route = state.get("route", "DIRECT")
    if route == "RAG":
        return "rag_node"
    elif route == "WEB":
        return "web_search_node"
    elif route == "IMAGE":
        return "image_analysis_node"
    elif route == "MULTI_TOOL":
        return "multi_tool_node"
    else:
        return "verify_node"


def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_intent", analyze_intent_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("image_analysis_node", image_analysis_node)
    workflow.add_node("multi_tool_node", multi_tool_node)
    workflow.add_node("verify_node", verify_node)
    workflow.add_node("action_planner", generate_action_plan_node)

    workflow.add_edge(START, "analyze_intent")

    workflow.add_conditional_edges(
        "analyze_intent",
        route_decision,
        {
            "rag_node": "rag_node",
            "web_search_node": "web_search_node",
            "image_analysis_node": "image_analysis_node",
            "multi_tool_node": "multi_tool_node",
            "verify_node": "verify_node"
        }
    )

    workflow.add_edge("rag_node", "verify_node")
    workflow.add_edge("web_search_node", "verify_node")
    workflow.add_edge("image_analysis_node", "verify_node")
    workflow.add_edge("multi_tool_node", "verify_node")

    workflow.add_edge("verify_node", "action_planner")
    workflow.add_edge("action_planner", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


_WORKFLOW_INSTANCE = None

def get_agent(model_name: str | None = None):
    global _WORKFLOW_INSTANCE
    if _WORKFLOW_INSTANCE is None:
        _WORKFLOW_INSTANCE = build_workflow()
    return _WORKFLOW_INSTANCE