import cv2
import numpy as np

def process_pill_image(image_path):
    """
    [오토 줌 및 1:1 정사각형 타이트 크롭 함수]
    격자 트레이 사진에서 알약을 찾아내어 정중앙에 배치하고,
    가장 긴 변을 기준으로 1:1 정사각형으로 잘라낸 뒤 224x224로 변환합니다.
    """
    # 1. 원본 컬러 이미지 읽기 (BGR 컬러 유지)
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 이미지를 불러올 수 없습니다. 경로를 확인하세요: {image_path}")
        return None

    # 2. 계산을 위해 흑백 이미지(Grayscale)로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 이진화 (검은색 배경 0, 알약 255)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. 그림자 때문에 테두리가 파먹히는 현상을 막기 위한 팽창(Dilate) 연산
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # 5. 하얀색 덩어리(알약)의 외곽선(Contours) 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("❌ 알약을 찾을 수 없습니다. (빈 격자이거나 너무 어둡습니다)")
        return None

    # 6. 가장 큰 덩어리(알약) 고르기 및 기본 Bbox 추출
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # ==========================================
    # ✨ [핵심] 1:1 정사각형 구도 및 오토 줌 로직 적용
    # ==========================================
    # (1) 알약의 정중앙 좌표(Center) 계산
    cx = x + (w // 2)
    cy = y + (h // 2)

    # (2) 가로(w)와 세로(h) 중 '가장 긴 변'을 선택하여 정사각형의 한 변으로 삼음
    max_side = max(w, h)

    # (3) 알약 모서리가 잘리지 않도록 여유 공간(Padding)을 넉넉히 부여
    # 짧은 변 쪽에는 이 과정과 비례하여 자연스럽게 검은 트레이 여백이 남게 됩니다.
    padding = 5
    box_size = max_side + (padding * 2)

    # (4) 중심점을 기준으로 1:1 정사각형의 시작점과 끝점 계산
    half_box = box_size // 2
    x1 = max(0, cx - half_box)
    y1 = max(0, cy - half_box)
    x2 = min(img.shape[1], cx + half_box)
    y2 = min(img.shape[0], cy + half_box)

    # 7. 원본 컬러 이미지에서 정중앙 기준 1:1 정사각형 크롭!
    square_crop_img = img[y1:y2, x1:x2]

    # 8. AI 모델의 표준 규격인 224x224로 리사이징 (찌그러짐 없음)
    final_ai_ready_img = cv2.resize(square_crop_img, (224, 224))

    print(f"✅ 오토 줌 크롭 성공! 1:1 정사각형 크기: {box_size}x{box_size} -> 모델 입력 규격: 224x224")
    
    return final_ai_ready_img, square_crop_img

# ==========================================
# 🧪 테스트 실행 코드
# ==========================================
if __name__ == "__main__":
    test_image_file = r"C:\Users\go\Desktop\HOP project\open_cv_test\test_orbit.png" 
    
    # 함수 실행
    result = process_pill_image(test_image_file)
    
    if result is not None:
        final_img, original_square_crop = result
        
        # 화면에 결과 띄워보기
        cv2.imshow("1. 1:1 Square Crop (Auto-Zoom)", original_square_crop)
        cv2.imshow("2. AI Ready (224x224 Resized)", final_img)
        
        print("아무 키나 누르면 창이 닫힙니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()