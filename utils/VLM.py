import json
from langchain_core.messages import HumanMessage, SystemMessage

VLM_SYSTEM_PROMPT = (
    "You are an expert robotic vision agent. You will receive an overhead image of a workspace. "
    "Calculate the EXACT DEAD-CENTER of the requested object(s). "
    "Return the object coordinates as normalized floats (0.0 to 1.0). "
    "You MUST output ONLY a valid JSON object with NO markdown formatting and NO conversational text."
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