import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
c = Groq(api_key=os.environ["GROQ_API_KEY"])

for model in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
    t0 = time.time()
    try:
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Return JSON: {\"status\": \"ok\"}"}],
            response_format={"type": "json_object"},
            timeout=15.0
        )
        print(f"[{model}] SUCCESS in {time.time()-t0:.2f}s: {r.choices[0].message.content}")
    except Exception as e:
        print(f"[{model}] FAILED in {time.time()-t0:.2f}s: {e}")
