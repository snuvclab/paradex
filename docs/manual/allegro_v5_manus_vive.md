# Allegro V5 B + MANUS / VIVE DexPilot

오른손 MANUS의 손끝 위치를 DexPilot으로 16개 Allegro V5 관절 각도로 변환한다.
VIVE는 기존 `ViveManusROSReceiver`에서 손목 자세에 합성되고 xArm의 상대 이동을 제어한다.
새끼손가락은 Allegro에 대응 관절이 없어 최적화 대상에서 제외한다.

현재 mainpc의 드라이버 실행 설정은 `HAND:=right TYPE:=B NAMESPACE:=right`이다.
`rsc/robot/allegro_v5/allegro_right_B_kinematics.urdf`는 설치된 V5 B 드라이버 URDF의
링크와 관절을 그대로 추출한 모델이다. 시각화/충돌/관성 요소만 제거했고 원본 SHA256을
파일에 기록했다. A 모델과 관절 범위 및 손가락 길이가 다르므로 B 모델로 최적화한다.
출력은 드라이버 순서(검지 0–3, 중지 4–7, 약지 8–11, 엄지 12–15)의 rad 값이다.

## 실행

Allegro 드라이버를 기존 장비 절차대로 먼저 실행하고
`/right/allegroHand_0/joint_states`에 실제 메시지가 나오는지 확인한다.
현재 PC에서는 별도 터미널에서 다음 실행기를 사용할 수 있다.

```bash
~/.local/bin/start-allegro-ros
```

이 실행기는 `HAND:=right TYPE:=B NAMESPACE:=right`로 드라이버를 시작한다.
`capture_robot.py`는 드라이버를 자동 실행하지 않는다. MANUS 수신 시작 로그만
나오는 경우에도 Allegro 연결이 완료됐다는 뜻은 아니다. 관절 피드백이 없으면
`Waiting for allegro_v5 feedback on /right/allegroHand_0/joint_states`를 출력하고
피드백이 들어올 때까지 손 명령을 보류한다.

mainpc의 서로 다른 터미널에서 입력 발행기를 실행한다.

```bash
~/vive-teleop/scripts/manus_publish.sh
```

```bash
~/vive-teleop/scripts/ros2_publish.sh
```

장갑 전원, 동글 연결 및 개인 MANUS 캘리브레이션을 확인한다.
발행기는 `~/vive-teleop/config/manus/{left,right}.mcal`을 사용하며
캘리브레이션은 기존 `~/vive-teleop/scripts/manus_calibrate.sh`로 수행한다.

명령 발행 없이 실제 입력, 최적화 출력 및 Allegro 피드백을 확인한다.

```bash
conda activate robot
sis
cd ~/paradex
python -m tools.allegro_v5_teleop_check --duration 15
```

이 진단은 손 관절/라이브러리 명령 publisher를 만들지 않는다.
종료 코드 0은 합성된 MANUS/VIVE 프레임으로 명령을 계산했고 Allegro 피드백도 받았다는 뜻이다.
입력 또는 피드백이 없으면 진단 정보를 출력하고 코드 2로 종료한다.
`--no-vive`를 주면 MANUS만 검사한다.

실제 xArm + Allegro 텔레옵:

```bash
conda activate robot
sis
cd ~/paradex
python -m src.dataset_acquisition.hri.capture_robot \
  --device vive --hand-side right --arm xarm --hand allegro_v5 \
  --allegro-retarget dexpilot --scale 1.15 \
  --no-camera --name allegro_v5_dexpilot
```

기존 왼쪽 MANUS 제어 제스처로 동작/일시정지를 선택한다. VIVE 사용 시 왼쪽 장갑 입력도
필요하다. `c`는 기록 시작, `s`는 기록 중지, `q`는 종료이다. 기록하지 않는 대기 상태에서도
텔레옵은 동작한다. 카메라 기록은 `--no-camera`를 제거하면 기존 경로를 사용한다.

손만 움직이려면 `--arm none`으로 실행한다. VIVE도 제외하려면 추가로 `--no-vive`를 준다.
오른손 MANUS만으로 손을 제어하는 명령은 다음과 같다. 왼손 장갑 및 제어 제스처는
요구하지 않으며, `c`/`s`는 녹화만 시작/중지한다. `--pedal`을 추가하면 중앙 페달을
누르는 동안만 손이 움직인다.

```bash
python src/dataset_acquisition/hri/capture_robot.py \
  --name allegro_hand_only --hand-side right --hand allegro_v5 \
  --no-camera --arm none --no-vive
```

`--scale`은 손 크기 비율이다. 기본 1.15에서 손끝 도달 거리에 맞게 조정한다.
`--allegro-command-rate-hz`의 기본값은 30 Hz이다.
`--allegro-command-rate-hz 0`이면 capture의 목표 갱신 제한을 해제하여 매 teleop
루프에서 최신 목표를 전달한다. Allegro ROS 명령 발행 주기(100 Hz)는 바뀌지 않는다.
양손 모드에서는 함께 전달하는 왼손 목표의 갱신 제한도 해제된다. 이 옵션은
매핑을 되돌리거나 실제 센서/통신/모터 지연을 제거하지 않으며, 기존 기본 동작은 유지한다.
`--allegro-teleop-log`는 기존 진단 NPZ 기록을 활성화한다.
기존 직접 각도 매핑은 `--allegro-retarget direct`로 선택하며 이것이 기본값이다.

## 구현 및 검증

DexPilot은 별도 Python 프로세스로 실행되고 동시에 하나의 요청만 처리한다.
완료된 최신 명령을 반환하며 첫 입력과 오래 일시정지한 뒤의 재개에서는 새 결과를 기다린다.
유한하지 않거나 퇴화한 손목 랜드마크, 작업 프로세스 종료 및 1초 넘게 갱신되지 않은
결과는 예외로 처리한다. 결과는 V5 B URDF의 관절 제한으로 자른다.
세션 종료 시 작업 프로세스도 종료한다. 왼손 V5 모델은 지원하지 않는다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_allegro_v5_dexpilot.py tests/test_vive_manus_receiver.py \
  tests/test_capture_bimanual_state.py tests/test_capture_robot_allegro_alignment.py \
  tests/test_allegro_realtime_visualization.py
```

## 연결 후 실입력 검증 (2026-10-17)

Manus/Vive 합성 입력 391프레임을 15초 동안 처리했고 Allegro 피드백도 수신했다.
왼쪽/오른쪽 MANUS 유효 메시지는 1410/1427개, 무효 메시지는 0개였다.
DexPilot 호출 지연 p95는 7.96ms였다. 사용 중인 드라이버의 B 모델 관절 정의도 일치한다.
이 검증은 로봇 명령을 발행하지 않았으며 실제 동작 추종 정확도는 포함하지 않는다.

## 손가락 뿌리 굽힘 보정 (2026-10-17)

V5 B의 palm_link 원점은 해부학적 손목이 아니라 손가락 뿌리 근처에 있다.
MANUS 손목-손끝 벡터를 그대로 대응하면 사람의 손바닥 길이가 목표에 더해져
뿌리 관절을 편 상태로 끝관절 위주로 굽히는 해가 나올 수 있었다.
이제 사람의 중지 MCP를 로봇 중지 MCP의 중립 위치에 맞춘 뒤 벡터를 구성한다.
손끝 사이 벡터와 DexPilot 집기 제약은 보존한다. 기존 실행 옵션으로 적용된다.

동일한 실제 입력을 재생했을 때 검지/중지/약지 MCP 목표가 각각
-0.6/57.1/-12.0도에서 72.2/94.5/67.3도로 바뀌었다. 관절 제한은 유지한다.
수정 후 관련 테스트 89개 통과, 실제 합성 입력 217프레임 처리 및 Allegro 피드백 수신을
확인했다(8초, 호출 지연 p95 6.25ms). 실제 움직임에 대한 추종 검증은 포함하지 않는다.
이미 실행 중인 텔레옵 프로세스에는 반영되지 않으므로 재실행해야 한다.

## 집기 제약 활성 거리: 모든 손끝 쌍 2cm 미만 (2026-10-17)

엄지-검지/중지/약지와 검지-중지/약지, 중지-약지의 여섯 쌍 모두
엄격히 0.02m 미만일 때만 새 집기 제약을 활성화한다. 정확히 2cm에서는 활성화하지 않는다.
기존 엄지 쌍의 해제 조건은 5cm 초과로 유지하므로, 이미 켜진 제약은
2~5cm 구간에서 유지될 수 있다. 나머지 손가락 쌍은 기존처럼 두 손가락이 모두
엄지 집기 상태인 경우에만 제약을 적용하며, 자체 거리 조건도 2cm 미만으로 변경했다.

라이브러리의 나머지 손가락 쌍 조건에는 3cm가 고정되어 있어,
Allegro 전용 optimizer 어댑터를 추가했다. 원래 목적함수와 미분은 유지한다.
설치된 dex-retargeting 패키지 및 다른 로봇의 설정은 수정하지 않았다.
19.9/20/20.1/29mm 경계값을 여섯 쌍 모두 검사했고 기존 회귀 검증을 포함해
118개 테스트가 통과했다. 텔레옵을 재실행해야 새 설정이 반영된다.

## Unimanual 중앙 페달 제어

기존 실행 명령에 `--pedal`을 추가하면 중앙 페달을 누르는 동안 팔과 손의 텔레옵을
함께 활성화하고, 놓거나 장치가 분리되면 정지한다. 시작 시에는 정지 상태이다.
페달 모드에서는 왼쪽 MANUS 장갑의 제어 제스처를 사용하지 않고, 왼쪽 장갑 입력도
요구하지 않는다. `--pedal` 없이 실행하면 기존 unimanual 제스처 제어를 유지한다.
Bimanual은 기존처럼 중앙 페달을 사용한다.

```bash
conda activate robot
sis
cd ~/paradex
python -m src.dataset_acquisition.hri.capture_robot \
  --device vive --hand allegro_v5 --allegro-retarget dexpilot \
  --arm xarm --pedal --no-camera --name allegro_teleop
```

페달을 놓으면 Allegro는 측정된 현재 관절 위치를 유지한다. 추적 입력이 잠시 없어도
놓임 상태는 처리한다. 다시 누르면 실제 현재 팔 자세를 기준으로 VIVE를 재설정하여
정지 중 손목 이동이 재개 시 팔 목표의 점프로 이어지지 않게 한다.
녹화 `c`/`s`와 종료 `q`는 기존 키보드 조작을 유지한다.
페달을 사용할 때 장치 열기에 실패하면 시작을 중단하며 제스처 모드로 자동 전환하지 않는다.

mainpc에서 USB 0fd9:0086 Elgato Stream Deck Pedal 1대, 펌웨어 2.0.4.0,
장치 열기/접근 권한 및 중앙 스위치 눌림·해제 3회를 확인했다.
관련 테스트 130개 통과. 입력 모니터링과 모의 제어 검증을 수행했고 로봇 명령은 발행하지 않았다.

### August 18 mapping comparison

Use `--allegro-retarget legacy-0818` to select the final August 18, 2026
Allegro V5 mapping (`fe18720cf`). The frozen module preserves that commit's
MANUS ergonomic mapping, transform-only fallback, thumb signs/gains and raw
output. In particular it does not apply the October 26 thumb opposition
remapping or physical clipping inside the retargeter. Existing controller
behavior, feedback readiness, command rate limits, arm mapping, camera startup
and split input/control loops still apply; this is a retargeter comparison,
not a checkout of the entire historical capture stack.

This selection replaces only `allegro_v5` retargeters. An Inspire hand in a
mixed bimanual session keeps its configured mapping. It also works with
`--no-vive`. The default remains `--allegro-retarget direct`; `dexpilot` stays
available separately. The implementation and numeric fixtures are frozen
from Git so runtime execution does not require Git history.

```bash
python src/dataset_acquisition/hri/capture_robot.py \
  --name 1102_legacy0818 --hand-side bimanual \
  --hand-left inspire_dftp --hand-right allegro_v5 \
  --allegro-retarget legacy-0818
```

### Linear thumb-tip mapping in direct mode

Allegro V5 `direct` defaults to `--allegro-thumb-distal-mapping linear` to map
MANUS `ThumbDIPStretch` degrees 1:1 to the last thumb joint angle. Only action
index 15 (`thumb_distal`) changes. Existing physical limits (-0.09 to 1.8 rad)
remain active. Transform-only input uses the signed local x rotation of
`thumb_proximal` to `thumb_distal`. This option leaves `legacy-0818`, DexPilot
and other hand models unchanged.

Select `--allegro-thumb-distal-mapping sine` to restore the previous ergonomic curve
`1.2 * 1.65 * sin(radians(ThumbDIPStretch))`, clipped to the same limits.
Both capture and the simulation UI accept the same option:

```bash
python -m paradex.retargetor.experiment.allegro_retargeter_alignment_ui \
  --live --simulate --hand allegro_v5 \
  --allegro-thumb-distal-mapping linear
```

Capture and the simulation UI use linear mapping when the curve option is
omitted. `--allegro-retarget direct` remains the default retargeter.
