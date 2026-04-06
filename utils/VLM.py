import json
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import (HF_API_TOKEN, HF_BASE_URL, GOOGLE_API_KEY, SELECTED_VLM)

VLMS = {
    1: {"label": "Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud"},
    2: {"label": "gemini-2.5-flash"},
}

MODEL_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 512,
    "max_retries": 3
}

VLM_SYSTEM_PROMPT = (
    "You are an expert robotic vision agent. You will receive an overhead image of a workspace. "
    "Calculate the EXACT DEAD-CENTER of the requested object(s). "
    "Return the object coordinates as normalized floats (0.0 to 1.0). "
    "You MUST output ONLY a valid JSON object with NO markdown formatting and NO conversational text."
)

def SELECT_VLM():
    if SELECTED_VLM == 1:
        return ChatOpenAI(
            model=VLMS[SELECTED_VLM]["label"],
            api_key=HF_API_TOKEN,
            base_url=HF_BASE_URL,
            temperature=MODEL_CONFIG["temperature"],
            max_tokens=MODEL_CONFIG["max_tokens"],
            max_retries=MODEL_CONFIG["max_retries"],
        )
    else:
        return ChatGoogleGenerativeAI(
            model=VLMS[SELECTED_VLM]["label"],
            api_key=GOOGLE_API_KEY,
            temperature=MODEL_CONFIG["temperature"],
            max_retries=MODEL_CONFIG["max_retries"],
        )  

def ASK(llm, prompt_content):
    try:
        response = llm.invoke([
            SystemMessage(content=VLM_SYSTEM_PROMPT), 
            HumanMessage(content=prompt_content)
        ])
        
        raw = response.content.strip()

        print(f"\n[LLM] Raw Response: {raw}")
        
        # Robust parsing (ignores conversational text)
        start_idx = raw.find('{')
        end_idx = raw.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            raw = raw[start_idx:end_idx+1]
            return json.loads(raw)
        else:
            print(f" -> [!] No JSON found in response: {raw}")
            return {"found": False}
            
    except Exception as e:
        print(f" -> [!] API Error / Parsing Error: {e}")
        return {"found": False}