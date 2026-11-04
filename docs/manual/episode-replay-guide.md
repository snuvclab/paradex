# 초기 6D 추정 기반 dataset retrieval 및 episode replay

확인 기준: 2026-11-04, mainpc의 `~/paradex`, `episode_replay` 브랜치.
아래 설명은 코드 확인 결과이며, 문서 작성 중 카메라 촬영이나 로봇 실행은 수행하지 않았다.

## 코드 위치

모든 경로는 `~/paradex` 기준이다.

| 역할 | 파일 / 주요 함수 |
| --- | --- |
| 촬영 → 초기 6D 추정 → retrieval → replay 통합 진입점 | `src/object6d/capture_sil.py`: `run_round()`, `run_replay_idle_loop()` |
| 초기 물체 pose 기반 episode 검색 | `paradex/retargetor/experiment/replay_pose_retrieval.py`: `rank_by_object_pose()` |
| 최상위 episode 전체 궤적 계획, preview, 실행 연결 | 같은 파일의 `replay_closest_episode_naive()` |
| 물체 기준 팔 궤적 변환, 손 명령 리샘플링, preview와 실제 실행 | `paradex/retargetor/experiment/replay_object_relative.py`: `relative_arm_actions()`, `_zoh_resample()`, `_compose_replay_trajectory()`, `_execute()` |
| 초기 6D 추정 후 실시간 손목 자세 기반 grasp 후보 시각화·선택 및 replay | `paradex/retargetor/experiment/grasp_indexing_visualization.py`: `rank_episode_grasp_frame_matches()`, `LiveGraspIndexingVisualization`, `_run_selected_replay_cycle()` |
| 검색 없이 지정한 episode를 frame 0부터 재생 | `paradex/retargetor/experiment/replay_object_relative_episode.py` |
| 별도 retrieval trajectory 데이터셋의 Allegro V5 grasp 재생, cuRobo 접근 계획 | `src/object6d/replay_grasp_retrieval_allegro_v5.py` |

## 기본 파이프라인

1. `capture_sil.py`가 원격 카메라로 한 프레임을 촬영하고 calibration을 저장한다.
2. Object6D RPC 서버에 이미지 경로와 mesh 이름을 보내 초기 물체 6D pose를 받는다. 이 진입점의 기본 RPC 주소는 `tcp://192.168.0.3:5570`이며, 실제 서버 위치에 맞춰 `--rpc_addr`로 변경할 수 있다. 다른 진입점은 기본 주소가 다를 수 있다.
3. `pose_robot = inverse(C2R) @ pose_world`로 로봇 좌표계의 물체 pose를 계산한다.
4. 각 dataset episode의 첫 프레임 물체 pose와 현재 pose를 비교한다. 기본 검색 점수는 다음과 같고, 작은 값부터 선택한다.

   ```text
   score = sqrt((translation_error / 0.05 m)^2
              + (rotation_error / 0.25 rad)^2)
   ```

5. 최상위 episode의 팔 궤적 전체를 현재 물체 pose에 맞춰 변환한다.

   ```text
   delta = T_robot_current_object @ inverse(T_robot_source_object)
   T_robot_replay_eef[t] = delta @ T_robot_source_eef[t]
   ```

6. 손 명령을 팔 시각에 맞게 zero-order hold로 리샘플링하고, 현재 자세에서 episode 시작 자세로 접근하는 궤적을 붙인다. preview 후 `--execute`와 사용자 확인이 있으면 팔·손을 재생한다.

`grasp_indexing_visualization.py`의 검색 방식은 위 첫 프레임 pose 검색과 다르다. 초기 물체 pose를 한 번 추정한 뒤, 각 episode의 캐시된 grasp frame을 현재 물체 기준으로 배치하고 **현재 손목의 물체 상대 회전 오차**로 후보를 정렬한다.

## 데이터와 결과 경로

- 기본 dataset root: `~/shared_data/capture/eccv2026/v0`
- 기본 episode 구조: `<capture-root>/allegro_v5/<물체명>/<숫자 episode>/`
- `--capture-root`로 dataset root, `--episode-root`로 물체별 episode 디렉터리를 직접 지정할 수 있다.
- 자동 후보 검색은 `grasp_result.json`에 `grasp_success: true`인 episode가 하나라도 있으면 성공 episode만 사용한다. 성공 라벨이 없으면 모든 숫자 episode 디렉터리를 사용한다. `--candidate-episodes 0,1,2`로 명시할 수도 있다.
- `--replay --name banana`는 별도 기본값 `~/shared_data/capture/0825_test_4/banana`를 사용한다. 일반 ECCV 데이터셋은 `--naive-replay` 또는 명시적인 `--episode-root` 사용이 명확하다.
- 이번 촬영/추정 결과 기본 위치: `~/shared_data/object_6d/sil/<YYYYMMDD_HHMMSS>/`
- 주요 결과: `object_6d.json`, `C2R.npy`, retrieval-only의 `retrieval_result.json`, replay의 `replay/naive_relative_replay_plan.npz`.

## 실행 예시

mainpc에서 실행한다. 아래 `OBJECT`와 dataset 경로는 실제 mesh 이름과 데이터에 맞게 변경한다. 카메라 daemon, Object6D RPC 서버, calibration이 준비되어 있어야 하며, preview에는 현재 로봇 상태를 읽을 ROS 드라이버가 필요하다.

```bash
cd ~/paradex
conda activate robot
OBJECT=banana

# 사진 촬영 + 초기 6D 추정 + episode 순위 저장
python src/object6d/capture_sil.py --name "$OBJECT" --retrieval-only

# 검색 + 전체 episode 재생 계획/preview
python src/object6d/capture_sil.py --name "$OBJECT" --naive-replay

# 같은 경로에서 실제 재생 허용: 확인 프롬프트에서 y 입력 시 실행
python src/object6d/capture_sil.py --name "$OBJECT" --naive-replay --execute
```

Replay 모드는 idle 프롬프트에서 `c`로 촬영·계획 round를 시작하고 `q`로 종료한다. 기본적으로 실제 움직임에는 `--execute`가 필요하다. `--auto-execute`는 `--execute`를 포함하며 실행 확인을 생략한다.

다른 데이터 위치를 사용하는 예:

```bash
python src/object6d/capture_sil.py \
  --name "$OBJECT" --retrieval-only \
  --episode-root /home/temp_id/shared_data/capture/EXPERIMENT/allegro_v5/OBJECT \
  --rpc_addr tcp://192.168.0.3:5570
```

실제 재생을 사용할 때는 기존 teleop 등 팔·손 명령 publisher와 동시에 실행하지 않는다. 이 문서는 경로와 옵션을 설명하며 현재 RPC 서버/드라이버의 동작 상태를 보증하지 않는다.
