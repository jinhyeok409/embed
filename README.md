# Embed: Hand Gesture Controlled Smart Bulb & Fall Detection System

원격 CCTV 낙상 감지 시스템으로 확장 중인 임베디드 시스템 프로젝트입니다.

---

### 3️⃣ 원격 CCTV 및 낙상 감지
- **설명**: OpenCV와 MediaPipe Pose를 활용해 사람의 스켈레톤을 실시간 추적. 어깨선(Y축)이 급격히 내려가거나 특정 임계값 이하로 떨어지면 낙상으로 판단.
- **기능**: 낙상 감지 시 사진 캡처 후 지역 관리자의 전화번호로 긴급 메시지(SMS/이메일) 전송.
- **장점**: 비접촉 모니터링으로 노약자 안전 강화.
- **도전 과제**: 실시간 스켈레톤 추적의 정확도, 네트워크 지연 최소화.

### 3️⃣ 사회적 기여: 노약자 및 환자 지원
- **아이디어**: 손동작으로 조명 제어 + 낙상 감지로 몸이 불편한 사람들의 안전과 편의를 지원.
- **가치**: 비접촉 인터페이스와 긴급 상황 대응으로 삶의 질 향상.
- **확장**: 손동작 인식이 어려운 경우 표정 인식 등으로 보완 가능.

---

## 하드웨어 및 정보
- **전구 모델**: Tapo L530
- **프로세서**: Raspberry Pi 4
- **카메라**: GD-C100 (USB 카메라)
- **IP 주소**: 192.168.1.41 (라즈베리파이)

---

## 소프트웨어 사용법
1. 라즈베리파이 전원 연결.
2. VNC 뷰어에서 `pi` 계정으로 연결 (IP: `192.168.1.41`, 비밀번호: `ki258436`).
3. 터미널에서 다음 명령어 실행:
   ```bash
   cd /mediapipe
   source mediapipe_env/bin/activate
   python tapo_hand.py
   ```
4. 프로그램 실행 후 손동작으로 전구 제어.
5. 종료하려면 `q` 키 입력.

---

## 스토리 텔링
출처 : 질병관리청

미국에서는 65세 이상 노인 중 1/3 이상이 1년에 최소 한 번은 낙상을 경험한다고 합니다. 우리나라의 경우 2020년 노인실태조사에 따르면 65세 이상 노인중 7.2%가 지난 1년 동안 낙상을 경험했으며, 낙상횟수는 평균 1.6회로 나타났고 나이가 많을수록 낙상률이 높았습니다(65~69세 4.5%, 85세 이상 13.6%). 흥미로운 사실은 2017년 노인실태조사에서는 1년간 낙상 경험 노인이 15.9%였고, 낙상횟수는 평균 2.1회인 것을 보면 3년 사이에 노인에서의 낙상 횟수가 줄어든 긍정적 신호로 볼 수도 있으나, 2020년 2월부터 코로나바이러스감염증-19 대응단계를 최고 단계인 ‘심각’으로 상향하면서 일상생활에서 사회적 거리두기를 조치함으로써 노인들의 사회적 활동이 감소된 것이 영향을 주었을 것으로 보입니다. 

손상 때문에 입원한 환자 중 80%는 65세 이상 노인입니다. 노인의 경우 평균재원일수도 높아서 전체 손상환자의 평균재원일수가 2018년 기준 13일인데 비해 노인은 16일간 입원하는 것으로 조사되었습니다. 특히 70세 이상이 되면 추락이나 미끄러짐으로 인해 입원하는 환자 수가 급격히 증가합니다. 노인에서 가장 흔한 손상 원인은 추락이나 낙상(60.9%)인데, 이는 교통사고(19.1%)보다 3배나 높습니다. 노인들은 낙상으로 인해 사망하는 경우가 많습니다. 미국인의 사망원인을 분석한 미국 질병관리예방본부(Centers for Disease Control and Prevention, CDC)의 보고에 따르면, 낙상은 65세 이상의 노인에서 손상으로 인한 사망의 가장 큰 원인입니다. 낙상으로 인한 사망률(나이 고려)은 2012년에는 10만명당 55.3건에서 2021년에는 78.0건으로 무려 41%가 늘었습니다. 전 세계적으로 고령의 노인 비율이 높아지면서 노인의 낙상 위험성이 더욱 증가하는 추세입니다. 이처럼 낙상은 노년 생활의 가장 큰 위험요소 중 하나이며, 노인 비율이 빠르게 증가하는 우리나라에서는 앞으로 더욱 증가할 것으로 예상됩니다.

## 개발 계획
1. 노트북으로 낙상 사고 감지 프로그램 개발
2. 카카오톡 API로 메세지 전달 프로그램 개발
3. 프로토타입1 = 낙상 사고 감지 프로그램 + 카카오톡 메세지 전달 프로그램 연결 
4. 노트북에서 프로토타입 1 정상작동 확인
5. 라즈베리파이에 프로토타입 1 연결
6. 시스템 외관 디자인 및 적용.


## 낙상을 파악하는 방법론
- 단순 Y축 기준으로 낙상 파악
=> 공간에 따라 오차 가능성 존재
=> 3차원 좌표계 활용

- https://github.com/barkhaaroraa/fall_detection_DL?tab=readme-ov-file
- 선행 프로젝트 깃헙링크
- 위 프로젝트는 어깨 좌표 기준으로 4초마다 디텍션을 하며(경량화), 급작스러운 좌표의 변동이 있을 시
falling 으로 파악하는 알고리즘을 구현 
- 얼굴 인식으로 누구인지 파악하는 것도 구현

++ 얼굴을 인식하는 것도 좋을 듯
환자 관리 차원에서 누구 낙상환자 발생이러면 좋을 거 같다고 느낌
