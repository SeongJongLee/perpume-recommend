from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from deepface import DeepFace
import google.generativeai as genai

app = Flask(__name__)

# Google Gemini API 설정
genai.configure(api_key="AIzaSyDOWUcoNLFDDgIkG-I23aBZqHK7i0WrsnE")

def analyze_face_and_recommend_fragrance(frame):
    try:
        # 얼굴 분석
        analysis = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)

        # 얼굴 특징(벡터) 추출
        embedding = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)

    except Exception as e:
        print(f"🚨 얼굴 분석 오류: {e}")
        return None

    # 얼굴 분석 결과
    age = analysis[0]['age']
    gender = analysis[0]['gender']
    emotion = analysis[0]['dominant_emotion']
    face_vector = embedding[0]['embedding']  # 128차원 벡터

    print(age, gender, emotion,face_vector)

    # Gemini AI를 이용한 향수 추천
    prompt = f"""
    다음은 한 사람의 나이, 성별, 감정, 얼굴  특징 벡터야. 이를 기반으로 향을 추천해줘:

    - Age: {age}
    - Gender: {gender}
    - Emotion: {emotion}
    - Face Vector: {face_vector[:5]}

    이 정보를 참고해서 먼저는 솔직하게 얼굴벡터를 분석해서 생각나는 눈매, 코, 볼, 턱 등의 특징을 세세하게 말해주되 알기 쉽게 10줄 이상 자세히 말해줘 대신 벡터에 대한 부분은 언급하지 말아줘.
    1. 어울리는 여러 향을 추천해주고 추천 이유와 느낌을 자세히 설명해줘.
    2. 어울리는 향에 맞는 향수를 한국어로 몇 가지 추천해줘. 
    3. 나이와 성별은 언급하지 말고, 3번은 따로 말 안해줘도 돼
    """

    # 최신 Gemini 모델 사용
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)

    print("\n🎯 **향수 추천 결과** 🎯")
    print(response.text.strip())

    return response.text.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "이미지가 필요합니다."})

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    fragrance = analyze_face_and_recommend_fragrance(image)
    return jsonify({"fragrance": fragrance})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
