import os
import csv
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. 모델 및 체인 설정
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0) # 분석은 일관성이 중요하므로 0으로 설정
prompt = PromptTemplate.from_template(
    "다음 리뷰의 감정을 '긍정' 또는 '부정'으로 분류하고, 짧은 이유를 쓰세요.\n\n리뷰: {content}"
)
chain = prompt | llm | StrOutputParser()

def run_batch_analysis():
    # 2. 파일 읽기
    file_path = "reviews.txt"
    
    if not os.path.exists(file_path):
        print(f"❌ {file_path} 파일이 없습니다! 파일을 먼저 만들어주세요.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        reviews = f.readlines()

    print(f"🚀 총 {len(reviews)}개의 리뷰 분석을 시작합니다...\n")

    # 3. 반복문으로 분석
    #-- for 하나씩 처리--
    # for i, review in enumerate(reviews):
    #     review = review.strip() # 줄바꿈 제거
    #     if not review: continue
        
    #     print(f"[{i+1}/{len(reviews)}] 분석 중...")
    #     response = chain.invoke({"content": review})
    #     results.append(f"리뷰: {review}\n분석: {response}\n{'-'*30}")
    
    #-- batch(일괄처리) --
    inputs = []
    valid_reviews = []
    
    # 1. 텍스트 파일에서 읽어온 데이터를 체인(Chain)이 먹을 수 있는 형태로 가공하기
    for review in reviews:
        review = review.strip()
        if not review: continue
        
        # 📦 inputs: 프롬프트의 {content} 빈칸에 채워넣을 딕셔너리 형태
        # 예: [{"content": "정말 좋아요!"}, {"content": "너무 느려요..."}]
        inputs.append({"content": review})
        
        # 📝 valid_reviews: 나중에 결과와 짝지어줄 원본 텍스트 형태
        # 예: ["정말 좋아요!", "너무 느려요..."]
        valid_reviews.append(review)

    chunk_size = 5 # 한 번에 서버에 보낼 묶음(배치)의 크기
    final_results = [] # 최종 엑셀 파일에 들어갈 표 형태의 데이터를 모을 빈 바구니

    print(f"🚀 총 {len(inputs)}개의 리뷰를 {chunk_size}개씩 나누어 분석합니다...\n")

    # 2. 데이터를 5개씩 쪼개서(Chunking) 일괄 처리(Batch) 하기
    for i in range(0, len(inputs), chunk_size):
        
        # 슬라이싱([시작:끝])을 이용해 전체 리스트에서 5개씩만 뚝 잘라오기
        chunk_inputs = inputs[i : i + chunk_size]
        chunk_reviews = valid_reviews[i : i + chunk_size]
        
        print(f"📦 [{i+1} ~ {min(i+chunk_size, len(inputs))}] 번째 데이터 배치 처리 중...")
        
        # 🤖 5개의 질문을 한꺼번에 AI에게 던지고, 5개의 답변을 받아오기
        # 반환 예: batch_responses = ["감정: 긍정\n이유: ...", "감정: 부정\n이유: ...", ...]
        batch_responses = chain.batch(chunk_inputs)
        
        # 3. zip()을 이용해 원본 리뷰와 AI 분석 결과를 1:1로 짝지어 표의 한 줄(Row)로 만들기
        for review, response in zip(chunk_reviews, batch_responses):
            # 엑셀의 [A열, B열]에 들어갈 형태로 리스트로 묶어서 추가
            # 예: ["정말 좋아요!", "감정: 긍정\n이유: ..."]
            final_results.append([review, response])
            
        # 4. 서버 과부하(Rate Limit 429 에러) 방지를 위해 60초 쿨다운
        if i + chunk_size < len(inputs):
            print("⏳ 서버 속도 제한(Rate Limit) 방지를 위해 60초 휴식...\n")
            time.sleep(60)

    # 5. 최종 결과를 엑셀(CSV) 파일로 예쁘게 저장하기
    csv_path = "analysis_results.csv"
    # 한글 깨짐을 막기 위해 'utf-8-sig' 사용, 빈 줄이 띄워지는 것을 막기 위해 newline=""
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["리뷰 원본", "AI 분석 결과"]) # 엑셀 첫 번째 줄(헤더) 이름표 붙이기
        
        # ⚠️ 수정 포인트: 위에서 짝지어 모아둔 final_results를 넣어줍니다!
        writer.writerows(final_results) 
    
    print(f"\n✅ 분석 완료! '{csv_path}' 파일이 생성되었습니다.")

if __name__ == "__main__":
    run_batch_analysis()