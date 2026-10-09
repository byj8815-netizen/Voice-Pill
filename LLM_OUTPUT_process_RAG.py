import requests
import json
import itertools
from openai import OpenAI
from typing import List, Dict, Any

# ==========================================
# 1. 설정, API 키 및 엔드포인트 입력란
# ==========================================

# [입력란 1] OpenAI API 키
OPENAI_API_KEY = ""
# ⚠️ 주의: 반드시 'Decoding(디코딩)' 키를 복사해서 넣으세요!
PUBLIC_DATA_API_KEY = "이건 공공 데이터 포털"

# [입력란 3] e약은요(의약품개요정보) API 엔드포인트
E_DRUG_ENDPOINT = " e약은요 API 엔드포인트 "

# [입력란 4] DUR(의약품안전사용서비스) API 엔드포인트
DUR_ENDPOINT = "DUR API 엔드포인트"


# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# 2. 공공데이터 API 호출 함수 (e약은요, DUR)
# ==========================================
def get_drug_info(item_seq: str) -> str:
    """
    [e약은요 API] 개별 약품의 효능, 용법, 주의사항을 가져옵니다.
    """
    # Section 1에서 정의한 엔드포인트와 인증키를 그대로 가져와 사용합니다.
    url = E_DRUG_ENDPOINT 
    params = {
        "serviceKey": PUBLIC_DATA_API_KEY, 
        "itemSeq": item_seq, 
        "type": "json"
    }
    
    try:
        # ⬇️ 실제 API를 호출하려면 아래 주석을 해제하세요.
        # response = requests.get(url, params=params, timeout=5)
        # data = response.json()
        # 추출 로직: 효능효과(efcyQesitm), 용법용량(useMethodQesitm), 주의사항(atpnQesitm)
        
        # [MOCK DATA] 테스트를 위한 가상 응답
        mock_db = {
            "002824": "효능: 해열, 진통, 소염. 주의사항: 위장출혈 위험이 있으니 식후 복용 권장.",
            "049428": "효능: 파킨슨증후군 치료. 주의사항: 졸음 유발 가능성.",
            "003532": "효능: 류마티스 관절염, 말라리아 치료. 주의사항: 시력 저하 등 안과 검진 필요."
        }
        seq_num = item_seq.replace("K-", "") # "K-002824" -> "002824"
        return mock_db.get(seq_num, "상세 정보를 찾을 수 없습니다.")
    except Exception as e:
        return f"API 호출 오류 발생: {e}"

def check_dur_interaction(item_seq_a: str, item_seq_b: str) -> str:
    """
    [DUR API] 두 약물 간의 병용금기 여부를 확인합니다.
    """
    # Section 1에서 정의한 엔드포인트와 인증키를 그대로 가져와 사용합니다.
    url = DUR_ENDPOINT
    params = {
        "serviceKey": PUBLIC_DATA_API_KEY,
        "itemSeq1": item_seq_a,
        "itemSeq2": item_seq_b,
        "type": "json"
    }
    
    try:
        # ⬇️ 실제 API를 호출하려면 아래 주석을 해제하세요.
        # response = requests.get(url, params=params, timeout=5)
        # data = response.json()
        
        # [MOCK DATA] 테스트를 위한 가상 병용금기 결과
        return "병용금기 사항 없음 (안전)"
    except Exception as e:
        return f"DUR 검증 오류: {e}"

# ==========================================
# 3. 데이터 처리 및 RAG 프롬프트 생성 모듈
# ==========================================
def generate_safe_pill_response(vision_results: List[Dict[str, Any]], confidence_threshold: float = 0.5) -> str:
    """
    비전 모델 결과를 바탕으로 API 데이터를 수집하고 LLM 답변을 생성합니다.
    """
    detected_drugs = []
    uncertain_drugs = []
    
    # 3-1. 비전 결과 분류 (신뢰도 기준)
    for res in vision_results:
        if res["confidence"] >= confidence_threshold:
            detected_drugs.append(res)
        else:
            uncertain_drugs.append(res)
            
    if not detected_drugs and not uncertain_drugs:
        return "인식된 알약이 없습니다. 사진을 다시 찍어주세요."

    # 3-2. RAG를 위한 Context (문맥) 정보 수집
    context_lines = []
    
    # 확실한 약물 정보 수집
    context_lines.append("### 개별 약물 정보 (e약은요 기준)")
    for drug in detected_drugs:
        info = get_drug_info(drug["code"])
        context_lines.append(f"- {drug['name']} ({drug['code']}): {info}")
        
    # 불확실한 약물 정보 처리 (환각 방지)
    if uncertain_drugs:
        context_lines.append("\n### 인식 불확실 약물 (사용자 확인 필요)")
        for drug in uncertain_drugs:
            context_lines.append(f"- {drug['name']} (추정치. 인식률 {drug['confidence']:.2f}로 인해 확실하지 않음)")

    # DUR 병용금기 교차 검증 (확실한 약물들끼리만)
    context_lines.append("\n### DUR 약물 상호작용 (병용금기) 정보")
    if len(detected_drugs) >= 2:
        # 약물들의 모든 2개 조합(Pair) 생성
        pairs = list(itertools.combinations(detected_drugs, 2))
        for pair in pairs:
            interaction = check_dur_interaction(pair[0]["code"], pair[1]["code"])
            context_lines.append(f"- {pair[0]['name']} & {pair[1]['name']}: {interaction}")
    else:
        context_lines.append("- 단일 약물이거나 조합을 검사할 약물이 충분하지 않습니다.")

    rag_context = "\n".join(context_lines)

    # 3-3. LLM 프롬프트 구성 (System & User)
    system_prompt = """
    당신은 친절하고 전문적인 약사 어시스턴트입니다. 
    제공된 [공식 약물 데이터]만을 바탕으로 사용자의 질문에 답하세요.
    절대 본인의 지식으로 약효나 부작용을 지어내지 마세요(할루시네이션 엄격히 금지).
    
    규칙:
    1. 인식 불확실한 약물이 있다면, 반드시 사용자에게 "XX약은 인식이 불확실하여 잘못된 약일 수 있으니 복용 전 확인하라"고 경고할 것.
    2. DUR 정보에 병용금기가 있다면 강력하게 경고할 것.
    3. 일반인이 이해하기 쉬운 문장으로 요약하여 설명할 것.
    """
    
    user_prompt = f"""
    아래 [공식 약물 데이터]를 분석해서 약들의 효능과, 이 약들을 같이 먹어도 안전한지 설명해줘.
    
    [공식 약물 데이터]
    {rag_context}
    """

    # 3-4. LLM 답변 생성
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # 또는 gpt-3.5-turbo
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0 # 환각을 최소화하기 위해 온도를 0으로 설정
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {e}"

# ==========================================
# 4. 실행 테스트
# ==========================================
if __name__ == "__main__":
    # 첨부해주신 이미지의 모델 추론 결과를 리스트 딕셔너리 형태로 변환
    vision_model_output = [
        {"slot": 0, "class": 700, "code": "K-002824", "name": "신일이부프로펜정 400mg 10T", "confidence": 0.452}, # 불확실
        {"slot": 1, "class": 411, "code": "K-049428", "name": "퍼킨 씨알정 50-200MG(명인)", "confidence": 0.671},
        {"slot": 2, "class": 384, "code": "K-003532", "name": "옥시크로린정 200MG (경풍)", "confidence": 0.643}
    ]
    
    print("--- [데이터 수집 및 LLM 분석 시작] ---\n")
    # 신뢰도 임계값을 0.5로 설정하여 0.452인 신일이부프로펜을 필터링
    final_answer = generate_safe_pill_response(vision_model_output, confidence_threshold=0.5) 
    print(final_answer)