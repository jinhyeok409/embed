import asyncio
from tapo import ApiClient
from tapo.requests import Color  # Ensure that we import Color from the correct module

# Tapo 계정 정보
TAPO_USERNAME = ""
TAPO_PASSWORD = ""
TAPO_BULB_IP = ""  # 전구의 로컬 IP 주소

async def control_tapo():
    # Tapo API 클라이언트 생성
    client = ApiClient(TAPO_USERNAME, TAPO_PASSWORD)

    # Tapo 전구 연결
    device = await client.l530(TAPO_BULB_IP)

    # 전구 켜기
    await device.on()
    print("전구를 켰습니다!")

    # 밝기 조절 (0~100%)
    await device.set_brightness(50)
    print("밝기를 50%로 설정했습니다.")

    # 색온도 조절 (2700K)
    await device.set_color_temperature(2700)
    print("색온도를 2700K로 설정했습니다.")

    # 5초 후 색상 변경 (Chocolate)
    await asyncio.sleep(5)
    await device.set_color(Color.DarkRed)  # Color::Chocolate
    print("전구 색상을 Red으로 변경했습니다.")

    # 5초 후 색온도 변경 (6500K)
    await asyncio.sleep(5)
    await device.set_color_temperature(6500)
    print("색온도를 6500K로 변경했습니다.")

    # 5초 대기 후 전구 끄기
    await asyncio.sleep(5)
    await device.off()
    print("전구를 껐습니다!")

# asyncio 실행
asyncio.run(control_tapo())
