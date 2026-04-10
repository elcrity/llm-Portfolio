from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. FastAPI 앱(서버) 생성
app = FastAPI(title="AI 리뷰 분석기 API")

# 2. AI 엔진 세팅 (서버가 켜질 때 한 번만 조립해 둡니다)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# prompt = PromptTemplate.from_template(
#     "다음 리뷰의 감정을 '긍정' 또는 '부정'으로 분류하고, 짧은 이유를 쓰세요.\n\n리뷰: {content}"
# )

prompt = PromptTemplate.from_template(
    """사용자의 지시사항: {instruction}
    
    [시스템 필수 규칙]
    당신의 최종 답변 맨 앞에는 반드시 이 리뷰의 성격에 따라 [긍정] 또는 [부정] 이라는 태그를 고정으로 달아주세요.
    
    리뷰: {content}"""
)

chain = prompt | llm | StrOutputParser()

class ReviewRequest(BaseModel):
  instruction: str
  content : str

@app.post("/api/analyze")
async def analyze_review(request: ReviewRequest):
  result = chain.invoke({
        "instruction": request.instruction,
        "content": request.content
    })
  
  return {
    "original_review": request.content,
    "ai_result": result
  }