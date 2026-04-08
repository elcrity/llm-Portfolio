import streamlit as st
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 1. AI 엔진 세팅 (화면 그려지기 전에 미리 조립!) ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# prompt = PromptTemplate.from_template(
#     "다음 리뷰의 감정을 '긍정' 또는 '부정'으로 분류하고, 짧은 이유를 쓰세요.\n\n리뷰: {content}"
# )

prompt = PromptTemplate.from_template(
    "{instruction}\n\n리뷰: {content}") 

chain = prompt | llm | StrOutputParser()

# --- 2. 웹 화면(UI) 그리기 ---
st.title("🎬 AI 리뷰 감정 분석기")
st.write("환영합니다! 여기에 리뷰 파일을 올리면 AI가 분석해 줄 거예요. 🚀")

# 파일 업로드 칸 만들기
uploaded_file = st.file_uploader("리뷰가 담긴 텍스트 파일(.txt)을 올려주세요.", type=["txt"])

user_interaction = st.text_area(
    "AI에게 명령할 작업을 입력해주세요!",
    value="다음 리뷰의 감정을 '긍정' 또는 '부정'으로 분류하고, 핵심 이유를 짧게 쓰세요. 반드시 이 리뷰의 성격에 따라 [긍정] 또는 [부정] 이라는 태그를 고정으로 달아주세요."
)

# --- 3. 버튼이 눌렸을 때의 동작 ---
if st.button("분석 시작하기"):
    if uploaded_file is not None:
        
        # [핵심] 3-1. 업로드된 메모리상의 파일을 텍스트로 읽어오기
        string_data = uploaded_file.getvalue().decode("utf-8")
        reviews = string_data.splitlines() # 줄바꿈 기준으로 잘라서 리스트로 만들기

        # 3-2. 데이터 다듬기 (빈 줄 제거)
        inputs = []
        valid_reviews = []
        for review in reviews:
            review = review.strip()
            if not review: continue
            inputs.append({"instruction": user_interaction,
                           "content": review})
            valid_reviews.append(review)

        # 3-3. 분석 진행 (청킹 + 배치 처리)
        chunk_size = 5
        final_results = []
        
        # 안내 문구 띄우기
        with st.spinner("⏳ AI가 열심히 리뷰를 분석하는 중입니다..."):
          
          progress_bar = st.progress(0)
          
          for i in range(0, len(inputs), chunk_size):
              chunk_inputs = inputs[i : i + chunk_size]
              chunk_reviews = valid_reviews[i : i + chunk_size]
              
              # API 배치 호출
              batch_responses = chain.batch(chunk_inputs)
              
              for review, response in zip(chunk_reviews, batch_responses):
                  # 스트림릿의 표 만들기 기능을 위해 딕셔너리 형태로 묶어줍니다.
                  final_results.append({"리뷰 원본": review, "AI 분석 결과": response})
                  
              current_progress = min(i + chunk_size, len(inputs)) / len(inputs)
              progress_bar.progress(current_progress)
                  
              if i + chunk_size < len(inputs):
                  time.sleep(2) # 라이트 버전은 조금만 쉬어도 됩니다!

        # 3-4. 결과 출력
        st.success("✅ 분석 완료!")
        
         # ==========================================
        # 🌟 새롭게 추가되는 데이터 시각화(차트) 코드 🌟
        # ==========================================
        st.subheader("📊 리뷰 감정 요약 통계")
        
        # 1. 긍정과 부정 개수 세기
        positive_cnt = 0
        negative_cnt = 0
        for res in final_results:
            if "긍정" in res["AI 분석 결과"]:
                positive_cnt += 1
            else:
                negative_cnt += 1
                
        # 2. 화면을 반으로 나눠서 핵심 요약 숫자 보여주기 (st.metric 활용)
        col1, col2 = st.columns(2)
        
        col1.metric("긍정 리뷰", f"{positive_cnt}개")
        col2.metric("부정 리뷰", f"{negative_cnt}개")
        
        chart_data = pd.DataFrame({
            "개수" : [positive_cnt, negative_cnt]
        }, index=["긍정", "부정"])
        
        st.bar_chart(chart_data)
        st.divider()
        
        # 엑셀 파일 대신 스트림릿의 데이터프레임(표) 기능으로 화면에 바로 쏴주기!
        st.dataframe(final_results)
        df = pd.DataFrame(final_results)
        
        csv_data = df.to_csv(index=False).encode('utf-8-sig')
        
        st.download_button(
          label="📥 엑셀(CSV) 파일로 다운로드",
          data=csv_data,
          file_name="reviews.csv",
          mime="text/csv"
        )
        
        

    else:
        st.error("앗! 파일을 먼저 업로드해 주세요.")