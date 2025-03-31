async function setup() {
  // 모델 로딩 (최신 face-api.js 버전에서 사용하는 방식)
  await faceapi.nets.ssdMobilenetv1.loadFromUri('/models');
  await faceapi.nets.ageGenderNet.loadFromUri('/models');
  await faceapi.nets.faceExpressionNet.loadFromUri('/models');

  // 웹캠 연결
  const video = document.getElementById('webcam');
  const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
  video.srcObject = stream;

  video.onplay = () => {
    detectFace(video);
  };
}

async function detectFace(video) {
  // 얼굴 인식
  const detections = await faceapi.detectAllFaces(video)
    .withAgeAndGender()
    .withFaceExpressions();

  if (detections.length > 0) {
    const { age, gender } = detections[0];
    const { happiness, sadness, anger, surprise, neutral } = detections[0].expressions;

    // 나이, 성별, 감정 업데이트
    document.getElementById('age').textContent = Math.round(age);
    document.getElementById('gender').textContent = gender;
    const emotion = getEmotion(happiness, sadness, anger, surprise, neutral);
    document.getElementById('emotion').textContent = emotion;
  }

  // 주기적으로 얼굴 인식
  setTimeout(() => detectFace(video), 100);
}

function getEmotion(happiness, sadness, anger, surprise, neutral) {
  const emotions = [
    { emotion: 'Happy', value: happiness },
    { emotion: 'Sad', value: sadness },
    { emotion: 'Angry', value: anger },
    { emotion: 'Surprised', value: surprise },
    { emotion: 'Neutral', value: neutral }
  ];

  // 가장 높은 값의 감정 리턴
  const mostLikelyEmotion = emotions.reduce((prev, current) => (prev.value > current.value ? prev : current));
  return mostLikelyEmotion.emotion;
}

// 페이지 로드 후 실행
setup();
