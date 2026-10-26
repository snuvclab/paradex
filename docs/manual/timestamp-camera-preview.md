# Timestamp camera Web UI (22684253)

MacBook에서 mainpc의 timestamp 카메라를 필요할 때만 미리보기합니다.
영상은 저장하지 않으며 UTG 신호 발생기와 다른 카메라는 제어하지 않습니다.

## MacBook에서 사용

`/Users/taeyunha/mainpc_remote/camera-webui.command`를 Finder에서 더블클릭하거나
터미널에서 실행합니다.

```bash
/Users/taeyunha/mainpc_remote/camera-webui.command
```

브라우저 주소는 <http://127.0.0.1:8765>입니다.

1. **스트리밍 시작**: serial 22684253만 연결하여 기본 10fps, 최대 폭 960px로 표시합니다.
2. **중지 · 카메라 해제**: acquisition을 중지하고 변경한 카메라 설정을 복원한 뒤 연결을 해제합니다.
3. 브라우저 heartbeat가 약 15초 동안 없으면 자동 해제합니다. 백그라운드 탭의
   타이머가 제한되거나 Mac이 잠자기에 들어간 경우에도 자동 중지될 수 있습니다.

미리보기 중에는 해당 카메라를 점유합니다. Timestamp 수집을 실행하기 전에 반드시
중지하세요. 다른 프로그램이 이미 카메라를 점유하면 오류가 표시되며, 점유 프로그램을
강제로 종료하거나 카메라를 리셋하지 않습니다.

Web UI의 mainpc 수신 시각은 브라우저의 시간대로 표시합니다. Camera timestamp는
카메라 내부 클록의 나노초 값이며 UTC 시각이 아닙니다.

로컬 8765 포트가 사용 중이면 다음처럼 다른 포트를 지정할 수 있습니다.

```bash
PARADEX_LOCAL_PORT=8766 /Users/taeyunha/mainpc_remote/camera-webui.command
```

SSH 터널은 런처 종료 후에도 유지됩니다. 터널이 끊기면 런처를 다시 실행하세요.
Web UI가 서비스 재시작 후 연결 오류를 표시하면 페이지를 새로고침하세요.

## mainpc 서비스

구현은 `paradex/io/camera_system/timestamp_preview.py`와
`paradex/io/camera_system/templates/timestamp_preview.html`에 있습니다.
기존 `timestamp_monitor.py`와 `aravis_timestamp.py`는 수정하지 않습니다.

서비스는 HTTP를 `127.0.0.1:8765`에만 바인딩하고, Mac 런처가 SSH로 포워딩합니다.
서비스가 실행 중이어도 Start 전에는 카메라를 연결하지 않습니다. 부팅 자동 시작은
활성화하지 않았으며 런처가 필요할 때 서비스를 시작합니다.

현재 mainpc의 시스템 Python 3.8, Aravis 0.8, OpenCV 4.2, NumPy를 사용합니다.
Flask, Node, 브라우저 플러그인은 필요하지 않습니다.

```bash
# mainpc에서 최초 설치 (현재 계정의 repo 경로가 ~/paradex인 경우)
mkdir -p ~/.config/systemd/user
cp ~/paradex/systemd/paradex-timestamp-preview.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start paradex-timestamp-preview.service

# 상태 / 로그 / 서비스 중지
systemctl --user status paradex-timestamp-preview.service
journalctl --user -u paradex-timestamp-preview.service -n 50
systemctl --user stop paradex-timestamp-preview.service
```

시작 시 `TriggerSelector`, `TriggerMode`, `AcquisitionMode`, `PixelFormat`,
`AcquisitionFrameRateAuto`, frame-rate enable, `AcquisitionFrameRate`의 원래 값을
보관합니다. 중지·프레임 수신 오류·heartbeat 만료·정상 서비스 종료 시 복원합니다.
강제 전원 차단이나 SIGKILL은 정리 코드를 실행할 수 없습니다.
Gain, Exposure, IP, persistent user settings는 변경하지 않습니다.

## 검증

```bash
cd ~/paradex
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest -q tests/test_timestamp_preview.py
```

카메라 지연 연결, 중복 시작, 중지 및 해제, 초기화 실패 후 재시작, heartbeat 만료,
프레임 수신 timeout, 부분 설정 실패 시 복원, 구형 OpenCV에서의 Bayer 색 변환,
HTTP 제어 토큰 및 origin 검증, JPEG/MJPEG 전송을 테스트합니다.

실기 검증: 22684253에서 960×720 JPEG를 디코딩하고 프레임 증가를 확인했습니다.
중지 후 트리거와 frame-rate 설정의 전후 일치를 확인했습니다.
브라우저의 시각적 QA는 브라우저 접근 승인 거절로 수행하지 못했습니다.

API 참고: [Aravis buffer reference](https://aravisproject.github.io/docs/aravis-0.8/ArvBuffer.html),
[OpenCV Bayer conversions](https://docs.opencv.org/4.12.0/d8/d01/group__imgproc__color__conversions.html).
