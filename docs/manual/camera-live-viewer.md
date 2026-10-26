# 원격 카메라 실시간 뷰어

## 실행

현재 PC(`/home/taeyun/remote_mainpc`)에서:

```bash
./camera-viewer
```

브라우저에서 <http://127.0.0.1:8766>을 연다. 터미널을 유지하고, 종료할 때
Ctrl-C를 누른다. SSH 연결 종료 시 이 실행이 생성한 mainpc 뷰어도 종료된다.
capture PC daemon은 계속 동작한다. Python 표준 라이브러리만 사용하므로 설치가 필요 없다.

특정 카메라 바로가기:

<http://127.0.0.1:8766/?pc=capture16&camera=26256735>

카메라 선택은 PC별로 묶인 목록에서 한다. 기본 미리보기는 640×480, 최대 5 fps이며,
원본 선택 시 캐시된 2048×1536 JPEG를 최대 5 fps로 읽는다. 캡처 PC의 실제 설정이
다르면 해상도도 달라진다. 원본은 대역폭을 더 사용한다. 일시정지하거나 브라우저 탭이
숨겨지면 영상 요청을 멈춘다. 여러 창에서 같은 카메라·화질을 열면 mainpc가 요청을 합친다.

포트가 사용 중이면 **로컬과 원격 포트 모두** 바꿔 별도 실행할 수 있다:

```bash
PARADEX_VIEWER_LOCAL_PORT=8767 PARADEX_VIEWER_REMOTE_PORT=8767 ./camera-viewer
```

mainpc 자체에서 쓰려면:

```bash
cd ~/paradex
python3 src/camera/live_viewer.py --port 8766
```

기본으로 mainpc 및 현재 PC의 loopback에만 listen한다. SSH는 기존 `mainpc` 설정을
사용한다. LAN에 새 공개 포트를 열지 않는다. mainpc의 뷰어에는 필요시 `--config`,
`--daemon-port`, `--fps`(0 초과 10 이하), `--timeout`을 지정할 수 있다.

Mac 등 같은 Tailscale 네트워크에서도 접속하려면 기존 뷰어를 종료하고 실행한다:

```bash
./camera-viewer --tailscale
```

이 옵션은 mainpc의 현재 Tailscale IPv4 주소를 조회하고 뷰어를 그 주소에 bind한다.
2026-10-25 기준 Mac 접속 주소는 <http://100.124.206.14:8766>이다.
현재 PC의 <http://127.0.0.1:8766>도 SSH 터널을 통해 계속 사용할 수 있다.
Mac의 Tailscale 연결과 해당 기기에서 mainpc로 접근할 수 있는 tailnet 권한이 필요하다.
뷰어를 실행한 터미널을 유지해야 하며, Ctrl-C로 종료한다.

## 그리드 보기 (2026-10-26 추가)

기존 페이지를 새로고침하고 **보기 → 그리드**를 선택한다.
바로가기: <http://100.124.206.14:8766/?view=grid>.
현재 PC에서는 <http://127.0.0.1:8766/?view=grid>도 사용할 수 있다.

- 전체 카메라를 타일로 표시하며, 연결되지 않은 카메라는 상태를 표시한다.
- 카메라 그룹에서 Capture PC를 선택하거나 상단 PC 버튼을 누르면 해당 PC만 표시한다.
- 열 수는 자동/2/3/4/6열이다. 작은 모바일 화면에서는 읽기 쉽게 2열로 표시한다.
- 타일을 누르면 해당 카메라의 단일 보기로 확대한다. 보기 선택으로 그리드에 돌아온다.
- 그리드는 640px 미리보기 JPEG만 사용한다. 기본 2 fps이며 1/2/5 fps로 바꿀 수 있다.
  단일 보기에서는 기존 원본 화질 선택도 사용할 수 있다.
- `view`, `grid_pc`, `cols`, `grid_fps` 선택은 URL에 보존된다.
  예: `?view=grid&grid_pc=capture16&cols=2&grid_fps=2`.
- 브라우저 한 페이지에서 진행하는 영상 요청은 최대 4개이며, 카메라별로 순서대로
  조회 기회를 준다. 일시정지/숨겨진 탭에서는 진행 중 요청도 취소한다.
  그리드↔단일 전환이나 PC 필터 변경 때 제거된 카메라의 요청도 취소한다.
- 느린 카메라 하나가 나머지 조회를 멈추지 않도록 요청별 timeout과 재시도 간격을 둔다.
  기존 mainpc 캐시의 다중 창 요청 합치기도 유지한다.

그리드 배포 시 뷰어만 재시작했으며 capture daemon, 카메라, UTG 및 로봇 제어는
변경하지 않았다. 기존 촬영 전환의 일시적인 영상 정지 및 timestamp 한계는 그대로다.

## 확인한 시스템 구조

`system/current/pc.json`에 6개 capture PC와 카메라 시리얼이 지정되어 있다.
2026-10-25 실제 `/cameras` 응답 기준으로 23대가 연결되어 있다.

| PC | 제어망 IP | 연결 카메라 시리얼 |
| --- | --- | --- |
| capture12 | 192.168.0.12 | 23022639, 22684737, 23173282, 26256742 |
| capture13 | 192.168.0.13 | 23263780, 22684755, 23022632, 25452061 |
| capture14 | 192.168.0.14 | 23280285, 23012641, 22645026, 23280594 |
| capture15 | 192.168.0.15 | 23280286, 22645029, 23028333 |
| capture16 | 192.168.0.16 | 22641023, 22645021, 22641005, 26256735 |
| capture18 | 192.168.0.18 | 23263775, 23180202, 25452066, 25452062 |

capture15의 26256745는 설정에 있지만 현재 daemon에는 없다. 뷰어에서는 MISSING으로
표시하며 연결되지 않은 카메라를 다시 열거나 탐색하지 않는다.

capture PC SSH는 `paradex/io/capture_pc/ssh.py`에 정의된 포트 77을 사용한다.
각 PC의 `src/camera/server_daemon.py`가 Aravis/GStreamer backend로 실행 중이다.
daemon 한 프로세스가 해당 PC에 연결된 카메라들을 소유한다.

| 포트 | 역할 |
| --- | --- |
| 5480 | ZMQ ping/pong |
| 5481 | ZMQ PUB 상태/제어권 모니터 |
| 5482 | ZMQ REP 촬영 명령, register 및 heartbeat |
| 5483 | connection_port 필드에 예약됨; 조사한 daemon에서는 bind하지 않음 |
| 5484 | HTTP `/cameras`, `/preview/<serial>`, `/frame/<serial>` |

mainpc의 `remote_camera_controller`는 PC마다 독립 ZMQ worker를 두고 register로
제어권을 얻는다. prepare → start의 준비 완료 응답을 모은 뒤, 동기 촬영에서는
`CaptureSession`이 mainpc의 UTG900E 펄스를 켠다. 카메라는 Line0 rising edge를
받는다. controller는 0.5초 간격으로 heartbeat를 보내며, daemon은 명령 수신이
5초 끊기면 제어권을 해제하고 유휴 스트림으로 복귀한다.

Aravis가 얻은 원본 프레임은 GStreamer에서 Bayer 변환 후 두 경로로 나뉜다.
녹화 경로는 JPEG → AVI mux → 파일 저장이고, 미리보기 경로는 별도 leaky queue
(1 buffer) → 5 fps 제한 → 축소 → JPEG → appsink 캐시다.
원본 JPEG도 별도로 캐시한다. HTTP는 이 캐시를 읽는다.

```text
현재 PC 브라우저
  → localhost:8766 → SSH mainpc 터널
  → mainpc loopback:8766 (새 읽기 전용 gateway)
  → 선택된 capture PC:5484 (기존 daemon)
  → JPEG 캐시
```

## 캡처와의 동시 동작 및 한계

새 뷰어는 카메라 SDK, robot, UTG, remote_camera_controller를 import하지 않는다.
upstream으로 보내는 요청은 GET `/cameras`, `/preview/<serial>`, `/frame/<serial>`뿐이다.
촬영 제어용 ZMQ 포트에 접근하거나 register/prepare/start/stop/end를 보내지 않는다.
뷰어 종료, 연결 끊김, 실패한 HTTP 요청은 controller lease에 영향을 주지 않는다.
요청 대상은 설정의 PC·시리얼로 제한하며, 응답과 오류를 함께 캐시해 재시도 폭주를 막는다.

유휴 상태에서는 기존 daemon의 free-running stream을, 촬영 중에는 촬영 파이프라인의
동일한 프레임을 읽는다. 카메라 설정과 트리거 속도는 기존 캡처가 결정한다.
prepare/start/stop/end에서 파이프라인을 바꾸는 동안에는 잠시 영상이 중단될 수 있다.
트리거가 없는 동기 촬영 대기 상태에서는 새 프레임이 생성되지 않는다. 뷰어가 별도로
트리거를 켜지는 않는다. 프레임 카운터가 3초간 진행하지 않으면 갱신 대기를 표시한다.

기존 timestamp_preview(8765)는 다른 카메라를 직접 여는 별도 기능이다.
이번 기능은 capture PC daemon만 읽으며 timestamp 카메라와도 별개다.

## 변경 파일

mainpc `~/paradex`에 추가:

- `paradex/io/camera_system/live_viewer.py`: 읽기 전용 HTTP gateway
- `paradex/io/camera_system/templates/live_viewer.html`: 브라우저 UI
- `paradex/io/camera_system/templates/live_viewer.mjs`: 그리드/단일 보기 및 제한된 프레임 조회
- `src/camera/live_viewer.py`: checkout 실행 진입점
- `tests/test_live_viewer.py`: HTTP 통합 테스트
- `tests/test_live_viewer_grid.mjs`: 그리드 대상 선택·요청 제한·취소·timeout 테스트
- `docs/manual/camera-live-viewer.md`: 이 문서

현재 PC: `/home/taeyun/remote_mainpc/camera-viewer` 실행기 및
`paradex_viewer/` 아래 변경 소스·검증 기록 사본.
기존 수정사항은 유지했으며 capture PC 파일 변경이나 daemon 재시작은 없었다.

## 검증 (2026-10-25)

- 실제 SSH 터널을 통해 23대의 미리보기 JPEG를 각각 두 번 읽었고 모두 갱신됨.
- capture16이 idle이고 controller가 None인 것을 확인한 뒤 카메라 4대만
  free-running 30 fps로 3초 녹화. robot/UTG는 작동시키지 않음.
- 녹화 중 뷰어 원본 API에서 서로 다른 frame ID 12개 수신.
- 녹화 종료 후 idle/stream으로 복귀. 전환 중 HTTP 503 6회는 자동 복구됨.
- ffprobe로 MJPEG AVI 4개(2048×1536, 30 fps, 89~93 frames) 정상 확인.
  free-running 테스트이므로 카메라마다 시작 시점과 프레임 수는 다름.
- 새로운 HTTP 테스트 4개 통과: 촬영 상태 전환 중 조회, 다중 요청 합치기,
  실패 후 복구, 임의 대상/쓰기 거부, 오프라인 PC 격리.
- 관련 기존 테스트까지 62개 통과. 환경: `~/anaconda3/envs/robot/bin/python`,
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. 이 Conda Python이 `socket.IP_PKTINFO`를
  export하지 않아 기존 discovery 테스트 1개가 처음 실패했으며, 재실행 시
  테스트 프로세스 안에서 Linux 상수값 8만 보충함. 저장소 코드는 바꾸지 않음.
- Python 컴파일, UI JavaScript `node --check`, 실행기 `bash -n` 확인.
- SSH 연결 종료 시 새 gateway 종료/포트 해제 확인.
- 브라우저 자동 접근은 권한 거부로 차단되어 실제 브라우저에서 UI를 조작하는
  검증은 수행하지 못함. HTTP 및 JPEG 데이터 검증은 완료.

검증 JSON은 현재 PC의 `paradex_viewer/validation-20261025.json`과
`paradex_viewer/recording-validation-20261025.json`에 보관한다.
검증용 `/tmp/paradex-viewer-validation-20261025-155343`의 AVI 4개는 메타데이터를
보관한 후 정리했다.
외부 트리거 동기 캡처 전체 및 로봇 동작을 포함한 실험은 이번 검증 범위에 넣지 않았다.

테스트 재현(이 환경에서 필요한 상수 보충 포함):

```bash
cd ~/paradex
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/anaconda3/envs/robot/bin/python - <<'PY'
import socket, pytest
if not hasattr(socket, 'IP_PKTINFO'):
    socket.IP_PKTINFO = 8
raise SystemExit(pytest.main(['-q', 'tests/test_live_viewer.py',
    'tests/test_aravis_gstreamer_capture.py', 'tests/test_camera_daemon_reader.py',
    'tests/test_capture_pc_preview.py', 'tests/test_remote_camera_controller.py']))
PY
```

## 후속 사람 촬영 CLI 검증 (2026-10-25 16:07~16:09)

사용자가 지정한 사람 촬영 경로를 현재 저장소의 실제 경로인
`src/dataset_acquisition/hri/capture_hand.py`로 확인하고 `conda robot` 환경에서
수정하지 않은 CLI를 실행했다. `--name test --capture_root <고유 임시 경로>`를
사용했으며 c/s/q 명령을 stdin으로 전달했다. 각 회차는 약 5초간 기록했다.
23대 전체 및 mainpc timestamp 카메라, UTG 30 Hz 동기 촬영을 사용했다.

뷰어 프로세스는 공통으로 유지하고, 추가 영상 조회를 하지 않는 baseline과
capture16 / 26256735 원본 JPEG를 초당 5회 요청하는 경우를 두 번씩 비교했다.

| 비교 | 원본 조회 | 각 카메라 영상 프레임 수 (23대 모두 동일) | timestamp 수 |
| --- | --- | --- | --- |
| 1 | OFF | 155 | 155 |
| 1 | ON | 155 | 154 |
| 2 | OFF | 155 | 154 |
| 2 | ON | 155 | 155 |

- AVI 총 92개에서 ffprobe의 프레임 수 및 실제 packet 수가 모두 155로 일치했다.
  capture-PC 간 프레임 수 차이나 camera ERROR 상태는 관측되지 않았다.
- timestamp ID는 모든 회차에서 1부터 연속적이었고, 중간 누락·중복·역전은 없었다.
  다만 두 회차에서 영상보다 timestamp가 1개 적었다. 조회 ON/OFF에서 각각 한 번
  발생하여 뷰어에만 나타나는 문제가 아니다. 연속 ID가 154에서 끝난다는 점에서
  캡처 종료 경계의 마지막 timestamp가 저장되지 않는 문제로 추정되며, 원인 확정이나
  수정은 이번 테스트에 포함하지 않았다.
- 모든 회차의 첫 timestamp 간격은 47.90~48.21 ms였다. 이후 간격은 평균
  33.332~33.333 ms, 표준편차 0.032~0.045 ms였다. 이 시작 구간의 지연도 조회 여부와
  무관하게 나타났다. 저장 timestamp는 `time.time()` 기반 PC 수신 시각이므로,
  이 수치를 센서 하드웨어 클럭의 jitter 측정으로 해석하면 안 된다.
- 조회 ON 두 회차 모두 시작 약 0.037초 후 미리보기 API에서 HTTP 503이 한 번
  발생한 뒤 복구됐다. 촬영 파이프라인 전환 직후의 캐시 대기였다.
- 첫 비교가 끝난 뒤 piped stdin의 daemon 입력 스레드 때문에 Python 종료 오류가
  났다. 테스트 실행기에서 q 전달 후 stdin을 닫도록 바꿔 재실행했고 두 번째 비교는
  exit code 0, 오류 로그 없이 종료했다. 촬영 CLI/라이브러리 코드는 수정하지 않았다.

테스트 준비 중 별도로 확인한 환경/점유 문제:

- robot 환경에서 GI를 로드하려면 이 호스트에서 `GI_TYPELIB_PATH`와 시스템
  `libffi.so.7`을 지정해야 했다. 설정은 테스트 프로세스에만 적용했다.
- 10월 23일부터 남아 있던 ForgeHand recorder PID 3014751이 timestamp 카메라를
  점유하고 있었다. 카메라의 제어 포트 35792와 해당 프로세스의 UDP 포트가 일치했다.
  사용자의 명시적 승인 후 종료했다. SIGTERM으로 종료되지 않아 SIGKILL이 필요했고,
  카메라의 10초 heartbeat lease 만료 후 테스트를 재개했다.

결론: 이 짧은 테스트에서는 뷰어 조회에 따른 녹화 프레임 손실이나 timestamp 간격
악화를 관측하지 못했다. 기존 timestamp 기록에는 시작 간격 지연과 간헐적인
1-frame 끝 경계 불일치가 남아 있으므로, timestamp가 항상 영상과 정확히 일치한다고
판정할 수는 없다. 장시간/다중 카메라 동시 원본 조회 부하까지 검증한 결과는 아니다.

검증 후 6대 capture PC의 두 저장 드라이브에 생성한 임시 영상 약 21.8 GB,
mainpc shared_data의 timestamp 및 calibration 복사본, 실패한 시도의 임시 로그를
모두 삭제하고 각 경로가 없는 것을 확인했다. 6대 daemon은 idle/stream으로 복귀했다.
현재 PC `viewer_capture_check/`에는 비교 수치 JSON과 정리 내역 및 재현 스크립트만
남겼으며 녹화 영상이나 원본 timestamp 배열은 보관하지 않았다.

## 그리드 검증 (2026-10-26)

- 프런트엔드 Node 테스트 5개와 HTTP 통합 테스트 5개 통과.
- 실제 Tailscale 주소를 통해 그리드에서 사용하는 동일한 `FrameScheduler` 및
  `frameTargets` 코드로 23대의 미리보기를 조회했다. 각 카메라에서 최소 두 개의
  서로 다른 JPEG를 받았고, 총 53회 요청에 오류 0회, 최대 동시 요청 4개를 확인했다.
- 새 HTML/ES module의 HTTP 응답 및 JavaScript MIME type을 확인했다.
- 이번 변경에서는 촬영을 새로 시작하지 않았다. 그리드 전체 조회를 동반한
  동기 촬영의 frame loss/timestamp 재검사는 수행하지 않았다.
- 브라우저 UI 접근에 대한 이전 권한 거부는 우회하지 않았다. 따라서 실제 화면 배치나
  브라우저 내 클릭 조작은 직접 검증하지 못했으며 코드/요청 스케줄러/HTTP로 검증했다.

현재 PC의 검증 수치는 `paradex_viewer/grid-validation-20261026.json`에 보관한다.

```bash
node --test tests/test_live_viewer_grid.mjs
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_live_viewer.py
```
