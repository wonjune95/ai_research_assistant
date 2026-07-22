import os
import json
from dotenv import load_dotenv
from tavily import TavilyClient
load_dotenv()
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
# 검색 결과 가져오기
result = client.search("hello world test")
# json.dumps를 사용하여 들여쓰기(indent) 적용
print(json.dumps(result, indent=4, ensure_ascii=False))
