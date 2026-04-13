import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 1. AI 엔진 세팅 ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# 🌟 수정 포인트 1: AI가 여러 개의 리뷰를 한 번에 처리하도록 프롬프트 변경
prompt = PromptTemplate.from_template(
    """사용자의 지시사항: {instruction}
    
    [시스템 필수 규칙 - 절대 어기지 마세요]
    아래 제공된 {num_reviews}개의 리뷰를 한 번에 모두 분석해주세요.
    당신의 최종 답변은 반드시 각 리뷰 번호에 맞춰서 작성되어야 하며, 줄바꿈으로 구분해주세요.
    각 줄의 시작은 반드시 "번호. [긍정] 또는 [부정] 이유" 형식이어야 합니다.
    인사말이나 부연 설명 없이 사용자의 지시 사항에 맞춰 해당 내용을 출력해야합니다.
    
    리뷰 목록:
    {content}"""
) 

chain = prompt | llm | StrOutputParser()

# --- 2. 웹 화면(UI) 그리기 ---
st.title("🎬 AI 리뷰 감정 분석기")
st.write("환영합니다! 여기에 리뷰 파일을 올리면 AI가 분석해 줄 거예요. 🚀")

with st.expander("📝 텍스트 파일 작성 예시 (펼쳐보기)"):
    st.write("아래와 같이 한 줄에 하나씩 리뷰를 작성한 .txt 파일을 올려주세요.")
    st.code("""Zustand 라이브러리 덕분에 상태 관리 로직이 반으로 줄었어요. 개발 생산성 최고입니다.
새로 생긴 파스타집 갔는데 면이 다 불어서 나오고 소스도 너무 짜요. 다시는 안 갈 것 같네요.
이번 넷플릭스 신작 영화는 초반엔 흥미진진했는데 결말이 너무 허무해서 실망했습니다.""")
    st.info("💡 팁: 빈 줄 없이 작성하면 더 정확하게 분석됩니다.")

uploaded_file = st.file_uploader("리뷰가 담긴 텍스트 파일(.txt)을 올려주세요.", type=["txt", "csv"])

user_interaction = st.text_area(
    "AI에게 명령할 작업을 입력해주세요!",
    placeholder="ex)리뷰의 핵심 정보 요약해 줘, 리뷰의 핵심 키워드 뽑아줘"
)

# --- 3. 버튼이 눌렸을 때의 동작 ---
if st.button("분석 시작하기"):
    if uploaded_file is not None:
        
        string_data = uploaded_file.getvalue().decode("utf-8")
        reviews = string_data.splitlines() 

        # 데이터 다듬기 (빈 줄 제거)
        valid_reviews = [r.strip() for r in reviews if r.strip()]

        if not valid_reviews:
            st.warning("분석할 리뷰가 없습니다. 파일 내용을 확인해주세요.")
        else:
            # 🌟 수정 포인트 2: 리스트를 "1. 리뷰내용 \n 2. 리뷰내용" 형태의 하나의 긴 문자로 묶기
            combined_reviews = "\n".join([f"{i+1}. {review}" for i, review in enumerate(valid_reviews)])
            
            with st.spinner("⏳ AI가 리뷰를 분석하는 중입니다... (약 5~10초 소요)"):
                # 🌟 수정 포인트 3: for문 없이 단 1번만 AI 호출! (RPM/RPD 1만 소비)
                ai_response = chain.invoke({
                    "instruction": user_interaction,
                    "num_reviews": str(len(valid_reviews)),
                    "content": combined_reviews
                })
                
                # 🌟 수정 포인트 4: 한 덩어리로 온 답변을 다시 줄바꿈 기준으로 쪼개서 표에 넣기 좋게 매칭
                # AI가 준 답변을 한 줄씩 자름 (빈 줄 제외)
                response_lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
                
                final_results = []
                positive_cnt = 0
                negative_cnt = 0
                
                for i, review in enumerate(valid_reviews):
                    # 만약 AI가 실수로 결과를 덜 줬을 경우를 대비한 안전장치
                    if i < len(response_lines):
                        ai_result = response_lines[i]
                    else:
                        ai_result = "분석 누락"

                    final_results.append({"리뷰 원본": review, "AI 분석 결과": ai_result})
                    
                    if "[긍정]" in ai_result:
                        positive_cnt += 1
                    elif "[부정]" in ai_result:
                        negative_cnt += 1

            # --- 결과 출력 ---
            st.success("✅ 분석 완료!")
            
            st.subheader("📊 리뷰 감정 요약 통계")
            col1, col2 = st.columns(2)
            col1.metric("긍정 리뷰", f"{positive_cnt}개")
            col2.metric("부정 리뷰", f"{negative_cnt}개")
            
            chart_data = pd.DataFrame({
                "개수" : [positive_cnt, negative_cnt]
            }, index=["긍정", "부정"])
            st.bar_chart(chart_data)
            st.divider()
            
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