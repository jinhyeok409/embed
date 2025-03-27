import asyncio
from tapo import ApiClient

# Tapo 계정 정보
TAPO_USERNAME = ""
TAPO_PASSWORD = ""
TAPO_BULB_IP = ""  # 전구의 로컬 IP 주소

async def control_tapo():
    # Tapo API 클라이언트 생성
    client = ApiClient(TAPO_USERNAME, TAPO_PASSWORD)

    # Tapo 전구 연결 (P110 모델 대신 L530 등을 사용할 수도 있음)
    device = await client.p110(TAPO_BULB_IP)  # 또는 .l530(TAPO_BULB_IP) 사용

    # 전구 켜기
    await device.on()
    print("전구를 켰습니다!")

    # 5초 후 전구 끄기
    await asyncio.sleep(5)
    await device.off()
    print("전구를 껐습니다!")

# asyncio 실행
asyncio.run(control_tapo())
