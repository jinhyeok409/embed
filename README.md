# embed
1. 지금은 Ruled-base 기반으로 프로토 타입 제작
2. 추후 머신러닝으로 손동작을 인식하는 방향으로 진행

1️⃣ 기존 Rule-Based 방식 (현재 방식)

손가락 각도를 직접 계산해서 특정 패턴을 찾는 방법.
단점: 새로운 손동작을 추가하려면 일일이 수작업이 필요함.
장점: 가볍고 빠름.
2️⃣ 머신러닝/딥러닝을 활용한 방법 (추천)

데이터를 수집해서 학습시키는 방식.
손동작을 직접 분류하지 않고 AI가 알아서 패턴을 찾음.
장점: 새로운 손동작을 쉽게 추가할 수 있음.
단점: 데이터 수집과 모델 학습이 필요함.

## 확장성
; 우리 프로젝트의 가장 큰 장점! 확장성이 좋다 스마트 기기의 api만 따오면 라즈베리에서
컨트롤 가능!
1. 모델 확장
    - 현재는 코드로 규칙을 규정하여서 포즈를 분류하는 방식이다, 
    추후 파이토치등으로 모델자체를 학습시켜서 보다 정확한 손동작 인식 모델을 만들기
2. 앱 서비스 개발
    - 로그 관리 + 사용자 패턴 관리 등.
3. 맞춤형 손동작 서비스 구현?** --> 스토리 텔링 3까지 넘어감

## 스토리 텔링
1. 인간의 귀차니즘
    - 자려고 누웠는데 손동작 하나로 전구 컨트롤이 가능하면? 편리성 확보
    그러나 이미 음성인식 어플도 있고 그래서 애매하다.
2. 요리중 손이 뭐가 묻은 상태에서 손동작하나로 불켜기 
3. 몸이 불편한 분들에게 도움이 될 수 있다.
    -> 사실 이게 가장 좋긴한데 의미가 있어서 뭔가 모델 구성이 어려워질듯.
    손동작인식이 불능하면 표정인식등으로 도움이 될 수 있게 만들면 좋을 거 같은데,

## 정보
전구 모델은 L530, 라즈베리파이4 모델, 카메라 : GD-C100

## 소프트웨어 사용법 
1. 라즈베리파이 전원 연결.
2. VNC 뷰어에 이름 pi, 라즈베리파이 IP : 192.168.1.41 로 연결 (라베파 비밀번호 ki258436)
3. cd /mediapipe
4. source mediapipe_env/bin/activate 
5. python tapo_hand.py
6. 종료할꺼면 q
   



