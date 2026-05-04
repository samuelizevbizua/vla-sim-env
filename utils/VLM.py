import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import HF_TOKEN, HF_BASE_URL, VLM_PATH

MODEL_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 512,
    "max_retries": 3
}

VLM_SYSTEM_PROMPT = (
    "You are an expert robotic vision agent. You will receive an overhead image of a workspace. "
    "Calculate the EXACT DEAD-CENTER of the requested object(s). "
    "CRITICAL RULE FOR CONTAINERS: If asked to find a bin, crate, or container, you MUST return the coordinates for the empty hollow space in the very middle of the container, NEVER the outer plastic edges or rims. "
    "Return the object coordinates as normalized floats (0.0 to 1.0). "
    "You MUST output ONLY a valid JSON object with NO markdown formatting and NO conversational text."
)

LLM = ChatOpenAI(
    model=VLM_PATH,
    api_key=HF_TOKEN,
    base_url=HF_BASE_URL,
    temperature=MODEL_CONFIG["temperature"],
    max_tokens=MODEL_CONFIG["max_tokens"],
    max_retries=MODEL_CONFIG["max_retries"],
)

def ASK(prompt_content):
    try:
        response = LLM.invoke([
            SystemMessage(content=VLM_SYSTEM_PROMPT), 
            HumanMessage(content=prompt_content)
        ])
        
        raw = response.content.strip()
        
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