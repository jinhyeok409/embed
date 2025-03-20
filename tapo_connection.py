import asyncio
from kasa.discover import Discover

async def main():
    try:
        print("기기 검색 중...")
        devices = await Discover.discover()
        if not devices:
            print("발견된 기기 없음")
        for ip, device in devices.items():
            print(f"IP: {ip}")
            print(f"기기 정보: {device._info}")
            # device_type 대신 직접 확인
            print(f"모델: {device.model}")
    except Exception as e:
        print("오류 발생:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())