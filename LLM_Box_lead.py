import base64
from openai import OpenAI

# 1. API 키 설정 (보안을 위해 새로 발급받은 키를 넣으세요!)
API_KEY = "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"
client = OpenAI(api_key=API_KEY)

def encode_image(image_path):
    """로컬 이미지 파일을 Base64 문자열로 변환하는 함수"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ 이미지를 찾을 수 없습니다: {e}")
        return None

def extract_medicine_name_with_openai(image_path):
    """OpenAI Vision(gpt-4o)을 활용하여 약 상자의 제품명만 추출합니다."""
    
    # 이미지 인코딩
    base64_image = encode_image(image_path)
    if not base64_image:
        return None

    print("🤖 에이전트 AI: OpenAI 서버로 이미지를 전송하여 분석 중입니다...")
    
    prompt = """
    너는 의약품의 이름을 정확하게 식별하는 의료 AI 에이전트야.
    첨부된 사진은 약 상자 또는 약 포장지야. 
    사진 속 텍스트를 분석해서 이 약의 '가장 핵심이 되는 정확한 제품명'만 추출해.
    
    [제약 조건]
    1. 제조사명(예: 대웅제약, 종근당), 용량, 성분명, 주의사항은 절대 포함하지 마.
    2. 불필요한 서술어 없이 오직 '약 이름' 딱 하나만 단답형으로 출력해.
    3. 예시: '타이레놀정500밀리그램', '이지엔6이브연질캡슐', '판콜에이내복액'
    """

    try:
        # GPT-4o 모델 호출 (시각 분석에 가장 뛰어난 최신 모델)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=50 # 단답형이므로 길게 생성할 필요 없음
        )
        
        # 결과 텍스트 추출 및 여백 제거
        medicine_name = response.choices[0].message.content.strip()
        return medicine_name
        
    except Exception as e:
        print(f"❌ OpenAI API 호출 중 오류 발생: {e}")
        return None

# ==========================================
# 🧪 테스트 실행 코드
# ==========================================
if __name__ == "__main__":
    # 요청하신 타이레놀 사진 경로 (역슬래시 오류 방지를 위해 r을 붙인 raw string 사용)
    test_image_file = r"C:/Users/go\Desktop/HOP project/open_cv_test/타이레놀 검은 배경 데이터.jpg"
    
    # 1. 약 이름 추출 실행
    result_name = extract_medicine_name_with_openai(test_image_file)
    
    if result_name:
        print("\n========================================")
        print(f"✅ 인식 완료! 추출된 제품명 : [{result_name}]")
        print("========================================\n")
        
        # 2. 다음 단계 (에이전트 AI 상태 전이)
        print(f"🎙️ Voice Interface: \"일반 트레이에서 '{result_name}'을(를) 확인했습니다. 이 약이 맞으신가요?\"")