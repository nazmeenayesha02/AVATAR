import asyncio
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from agent import get_agent

async def test_scenarios():
    agent = get_agent("gemini-3.6-flash")
    config = {"configurable": {"thread_id": "hackathon_test"}}

    print("=== TEST 1: Greeting / Intro as Nazmeen Ayesha ===")
    inputs_1 = {
        "messages": [],
        "user_input": "hi who are you?",
        "language": "English",
        "image_path": None,
        "intent": "",
        "urgency": "NORMAL",
        "entities": {},
        "route": "DIRECT",
        "rag_evidence": "",
        "web_evidence": "",
        "image_evidence": "",
        "verification_notes": "",
        "action_plan": "",
        "sources": []
    }
    res_1 = await agent.ainvoke(inputs_1, config=config)
    last_msg_1 = res_1["messages"][-1].content
    print(last_msg_1[:300] + "...\n")

    print("=== TEST 2: Government Scheme Action Plan with Web Verification ===")
    inputs_2 = {
        "messages": [],
        "user_input": "I am a small farmer in Karnataka with 2 acres land. How do I get financial assistance under PM-Kisan and what documents are required?",
        "language": "English",
        "image_path": None,
        "intent": "",
        "urgency": "NORMAL",
        "entities": {},
        "route": "WEB",
        "rag_evidence": "",
        "web_evidence": "",
        "image_evidence": "",
        "verification_notes": "",
        "action_plan": "",
        "sources": []
    }
    res_2 = await agent.ainvoke(inputs_2, config=config)
    last_msg_2 = res_2["messages"][-1].content
    print(last_msg_2[:600] + "...\n")
    assert "WHAT I UNDERSTOOD" in last_msg_2 or "ACTION PLAN" in last_msg_2, "Missing action plan structure"

    print("=== TEST 3: Multilingual Action Plan (Kannada) ===")
    inputs_3 = {
        "messages": [],
        "user_input": "How to apply for caste certificate online in Karnataka?",
        "language": "Kannada",
        "image_path": None,
        "intent": "",
        "urgency": "NORMAL",
        "entities": {},
        "route": "WEB",
        "rag_evidence": "",
        "web_evidence": "",
        "image_evidence": "",
        "verification_notes": "",
        "action_plan": "",
        "sources": []
    }
    res_3 = await agent.ainvoke(inputs_3, config=config)
    last_msg_3 = res_3["messages"][-1].content
    print(last_msg_3[:600] + "...\n")

    print("=== ALL WORKFLOW TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    asyncio.run(test_scenarios())
