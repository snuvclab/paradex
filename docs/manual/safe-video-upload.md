# 검증 후 SSD 자동 정리 영상 업로드

2026-10-26 배포. 기존 capture-PC 업로더와 하위 rsync를 먼저 동결하고 종료한 뒤,
mainpc 및 capture12/13/14/15/16/18의 업로드 진입점을 안전한 구현으로 교체했다.
카메라/로봇 제어 프로세스는 중단하지 않았다.

2026-10-30 변경: 사용자 요청으로 정상 NAS 업로드 후 SSD 원본 자동 삭제를 기본값으로 복원했다.
실패·복구 영상은 보존한다. `--keep-raw`를 주면 이전처럼 원본을 보존한다.
현재 진행 중이던 6대 업로더는 정상 종료 후 재시작하며, 완료 파일은 기존 검증 기록과
NAS SHA-256을 확인하여 재인코딩 없이 정리한다. 카메라·로봇 데몬은 건드리지 않는다.

## 동작

1. `captures1`, `captures2`의 `**/raw/videos/*.avi`를 읽는다. 업로드 전에는 원본을 이동·수정·삭제하지 않는다.
2. 임시 출력은 항상 `~/captures1/.paradex_safe_upload/<run>/`에 만든다.
   동시에 2개만 처리하고 원본 크기의 두 배를 작업 공간으로 예약하며,
   추가로 20 GiB를 남기는 조건을 검사한다. 공간이 부족하면 실패 처리하고 원본을 보존한다.
3. ffprobe로 실제 입력 video packet 수를 세고, ffmpeg로 MJPEG(`-q:v 3`)를 만든다.
   해상도를 바꾸지 않고 `-vsync 0`으로 프레임 복제/제거를 방지한다.
4. 출력은 모든 프레임을 디코딩하며 입력 packet 수와 비교한다.
5. NAS의 숨김 임시 파일에 rsync하고, 로컬 출력과 NAS 파일의 SHA-256을 비교한다.
6. 검증된 파일만 최종 경로로 rename한다. 기존 NAS 파일이 있으면
   `<episode>/videos/.safe-upload-backups/<run>/`에 먼저 보존한다.
7. NAS 복사와 검증이 끝난 뒤에만 이 프로그램이 만든 임시 출력 파일을 정리한다.
8. 정상 업로드는 NAS 검증 파일이 바뀌지 않았고 원본 device/inode/size/mtime가 업로드 때와
   같은 경우, NAS 파일·디렉터리를 fsync한 뒤 SSD raw와 동일 회차의 오래된 처리본을 삭제한다.
   프레임 수·해상도·정상 목적지 매핑을 재확인하고, 열린 writer·최근 변경·복구 대상은 보호한다.
   삭제 직전 intent와 삭제 후 완료를 `<run>/deletions.jsonl`에 fsync하여 기록한다.
   NAS 파일·메타데이터·백업과 카메라 timestamp는 삭제하지 않는다.
   정리가 거부되면 `CLEANUP DEFERRED`와 `cleanup_error`를 남기며 SSD 원본을 보존한다.

검증된 정상 영상: `shared_data/<episode>/videos/<serial>.avi`.
파일마다 `<serial>.avi.safe-upload.json`에 프레임 수, 입력 정보, SHA-256, 경고 및
기존 NAS 파일 백업 위치가 기록된다.

## 손상 의심 입력

입력 AVI의 선언된 프레임 수가 없거나 실제 packet 수와 다르면 복구 대상으로 구분한다.
엄격한 디코딩에 실패하면 읽을 수 있는 프레임 복구를 시도한다. 입력/출력 개수가 다르거나
timestamp trigger span보다 영상이 유의하게 짧은 경우도 검토 대상으로 처리한다.

복구 영상은 `shared_data/<episode>/recovery/videos/<serial>.avi`에 저장한다.
정상 `videos` 파일로 가장하지 않으며 모니터에는 `failed` 및 `REVIEW REQUIRED`로 표시한다.
복구본도 출력 디코딩과 NAS 체크섬을 검증한다. 원본은 계속 보존한다.
전혀 읽을 수 없는 파일, 복구에 실패한 파일은 원본을 그대로 두고 오류 기록을 남긴다.

이전 점검에서 `261026_minicube_set6`의 capture12/18 원본 4개가 헤더 시각/프레임 수를
정상 제공하지 않았으나 시작 패킷은 읽혔다. 해당 회차 timestamp에는 frame ID
6994→7020 구간에서 25개가 빠져 있었다. 이 문제는 촬영 당시 생긴 문제로,
업로드가 이미 누락된 프레임을 복원하는 것은 아니다.

## 실행·모니터링

각 capture PC에서 기존 명령을 그대로 쓸 수 있다:

```bash
cd ~/paradex
conda activate flir_env
python src/util/upload_video/client.py --workers 2
```

로컬 전용 진입점도 같은 안전한 구현을 사용한다:

```bash
python src/validate/upload_raw_video/upload_local.py --workers 2 --no-publish
```

동일 PC의 uploader는 flock으로 중복 실행을 막는다. `SIGTERM`으로 중단하면
진행 중 인코딩을 중단하고 원본과 임시 전송 파일을 남긴다. 다시 실행할 때 이미
검증된 NAS 파일의 원본 signature와 SHA-256이 맞으면 재인코딩을 생략하고 정상 파일의 SSD 정리를 수행한다.
복구 대상의 REVIEW REQUIRED 상태는 재실행해도 유지한다.

기존 mainpc 모니터(`src/util/upload_video/process.py`, 포트 8081)에 진행 상태를
계속 PUB/SUB로 전달한다. 새 진입점은 기본 worker 수 2를 사용한다.

각 capture PC의 지속 기록:

- `~/.local/state/paradex/safe-upload/active-process.json`: 현재 PID와 로그 경로
- `~/.local/state/paradex/safe-upload/latest.json`: 최근 run 및 요약
- `~/.local/state/paradex/safe-upload/runs/<run>/progress.json`: 파일별 상태
- 같은 run 디렉터리의 JSON 및 ffmpeg 로그: 검증·오류 내역
- `~/.local/state/paradex/safe-upload/verified/`: 재실행용 검증 기록

배포 전 코드 백업은 각 PC의
`~/.local/state/paradex/safe-upload/code-backups/20261026T173306/before/`에 있다.
원본 보존이 필요할 때는 `--keep-raw`를 추가한다.

기존 `paradex/video/raw_video_processor.py`는 참조용으로 남아 있지만, 두 공식
업로드 진입점은 `safe_raw_video_processor.py`를 사용한다. 구형 함수를 직접 실행하면
이 안전장치를 우회하므로 사용하지 않는다.

## 검증

원본 보존 옵션, 자동 삭제, 검증된 기존 업로드 재개, rsync 실패, NAS checksum 불일치, 기존 NAS 파일 백업,
손상 헤더의 복구 경로 분리, 인코딩 프레임 부족, 열린 writer 보호, 삭제 직전 원본/NAS 변경 및 STOP 보호,
잘못된 회차·새 로컬 처리본 보호 테스트 17개 통과 (2026-10-30).
현재 PC와 mainpc robot Python에서 실행했다.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_safe_raw_video_processor_preservation.py
```

실제 capture-PC 업로드에서도 완료된 작업의 출력 프레임 수 및 NAS SHA-256 검증,
기존 모니터로의 상태 전달을 확인했다. 기본 모드에서는 검증 완료 후 SSD 영상이 자동 삭제되어
여유 공간이 늘어난다. `--keep-raw` 모드와 실패·복구 영상은 별도 정리 전까지 남는다.


## 업로드 완료 후 사용자가 요청한 SSD 정리

2026-10-26 업로드 완료 후 사용자가 failed 4개를 제외한 SSD 영상 삭제를 명시적으로 요청했다.
당시에는 일회성 `verified_ssd_cleanup.py --apply`로 정리했고 업로더의 원본 자동 삭제 정책은 바꾸지 않았다.
2026-10-30부터는 위 기본 자동 삭제 정책이 적용된다.

- 업로더의 공통 flock을 잡아 정리 중 새 업로드 작업이 동시에 시작되지 않게 한다.
- 정상 프레임 검증 기록 860개만 대상으로 하며, 복구·검토 대상 4개는 제외한다.
- NAS 영상 전체를 다시 SHA-256 계산하고 업로드 당시 기록과 일치하는지 확인한다.
- SSD 원본의 device/inode/size/mtime가 업로드 당시 signature와 같아야 한다.
- NAS와 SSD 파일이 검증 중 변경되거나 SSD 파일이 쓰기 모드로 열려 있으면 삭제하지 않는다.
- 동일 원본에서 생성된 예전 로컬 처리본도 정상 NAS 영상이 확인된 경우 정리한다.
- 실제 unlink 직전에 삭제 의도를 fsync한 journal에 기록하고, 삭제 후 완료 기록을 남긴다.
- 검토 대상 원본 4개의 signature는 정리 전후 다시 비교한다.

각 capture PC의 `~/.local/state/paradex/safe-upload/cleanup-latest.json`에서 정리 run을 찾는다.
해당 run 디렉터리에는 `plan.json`, `summary.json`, `journal.jsonl`이 남는다.
NAS 영상과 메타데이터, 백업 파일은 삭제하지 않는다.

진행 중 사용자 요청으로 전체 체크섬 재검증을 중단하고 `--folder-only` 모드로
전환했다. 이 모드는 기존 정상 업로드 기록과 동일 회차 NAS 폴더 존재를 기준으로
정리하며, 원본 signature 변경·열린 writer·복구 대상 제외 조건은 유지한다.
이 전환은 이번 정리에 대한 명시적 요청에 따른 것이다.

최종 정리 완료: 원본 860개 + 예전 처리본 120개 = 총 980개,
2,732,880,402,454 bytes(약 2.73 TB)를 삭제했다.
6대 모든 video 파일을 다시 검색하여 failed 원본 4개만 남았음을 확인했고,
4개 파일의 signature가 업로드 당시와 같은 것도 확인했다.
NAS 파일은 삭제하지 않았다. 삭제 journal 합계가 최초 SSD 목록의 파일 수·바이트 수와
정확히 일치했다. 최종 결과는 현재 PC `safe_upload/cleanup-result-20261026.json`에 있다.
