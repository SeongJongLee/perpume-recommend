import cv2
from flask import Flask, render_template, Response, jsonify
from deepface import DeepFace
import google.generativeai as genai

# Google Gemini API 키 설정
genai.configure(api_key="AIzaSyDOWUcoNLFDDgIkG-I23aBZqHK7i0WrsnE")

# Flask 애플리케이션 설정
app = Flask(__name__)


# 카메라 스트리밍을 위한 함수
def gen_frames():
    cap = cv2.VideoCapture(0)  # 웹캠을 열기

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 웹캠에서 프레임을 반환
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()


# 얼굴 분석 및 향수 추천 함수
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
    2. 어울리는 향에 맞는 향수를 몇 가지 추천해줘. 
    3. 나이와 성별에 대한 직접적인 언급은 하지 말아줘.
    """

    # 최신 Gemini 모델 사용
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)

    print("\n🎯 **향수 추천 결과** 🎯")
    print(response.text.strip())

    return response.text.strip()


# 메인 페이지 라우팅
@app.route('/')
def index():
    return render_template('index.html')


# 실시간 영상 스트리밍 라우팅
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 향수 추천 요청 처리
@app.route('/get_fragrance', methods=['POST'])
def get_fragrance():
    # 실시간으로 캡처한 프레임을 처리
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Failed to capture image"})

    # 분석하고 추천 받기
    fragrance = analyze_face_and_recommend_fragrance(frame)

    if fragrance:
        return jsonify({"fragrance": fragrance})
    else:
        return jsonify({"error": "Face analysis failed"})


if __name__ == '__main__':
    app.run(debug=True)
