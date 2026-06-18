# Procedural Pipeline (Unified)

Mario / Sokoban 두 게임을 한 폴더에서 공통 파이프라인으로 평가합니다.
공통 로직(G-Eval Auto-CoT, 20회 평균, CSV/Raw 저장, 분석 스크립트)은 모두 `procedural_pipeline/`에 있고, 게임별 차이(맵 파싱, 렌더러, 기본 경로)만 `procedural_pipeline/games/*.py`로 분리되어 있습니다.

## 폴더 구조

```
procedural_pipeline/
    judge.py                  # 공통 - G-Eval CoT, 프롬프트, OpenRouter 호출/파싱
    paths.py                  # 패키지/프로젝트 기준 경로 헬퍼
    execution/
        cli.py                # 단일 게임 실행 CLI
        experiments.py        # 세 실험 전체 실행 CLI
        runner.py             # 파이프라인 실행 루프
        csv_output.py         # 결과 CSV flatten/write
        options.py            # 실행 공통 옵션 기본값
    analysis/
        results.py            # 공통 분석 (--game ...)
        paper_proxy.py        # 논문 비교
        combined.py           # 두 게임 합친 논문 비교
        raw_runs.py           # raw JSON run-level 분석
        blind_ablation.py     # v1/v2 ablation 분석
        paper_v1_v2_table.py  # paper table 비교
        compare_original_maps.py
    games/
        mario.py              # 맵 정규화 + Mario jar 렌더
        sokoban.py            # 맵 정규화 + pcg_benchmark 렌더
    resources/
        mario/                # Mario 맵, 평가 기준, PlayAstar.jar, 렌더링 이미지
        sokoban/              # Sokoban 맵, 평가 기준
```

`third_party/pcg_benchmark`는 서드파티 코드라 이 폴더로 복사하지 않고, 기본 경로만 안정적으로 해석되도록 처리했습니다.

## 실행

세 실험 전체(Mario/Sokoban 모두):

```bash
uv run python -m procedural_pipeline.execution.experiments
```

실험별 실행:

```bash
# 1. 제작자 판단 + 자신감 + 5개 경험축
uv run python -m procedural_pipeline.execution.experiments --experiment 1

# 2. 제작자 질문 없이 5개 경험축
uv run python -m procedural_pipeline.execution.experiments --experiment 2

# 3. 모든 맵을 AI 제작 / Human 제작으로 각각 알려주고 5개 경험축
uv run python -m procedural_pipeline.execution.experiments --experiment 3
```

기존 `procedural_pipeline.run_experiments` 명령도 호환용 wrapper로 계속 동작합니다.
기본값으로 `execution.experiments`는 게임별 Auto-CoT 평가 스텝을 한 번만 생성해서 선택한 실험 전체에 재사용합니다.
즉 전체 실행 시 Mario 1회, Sokoban 1회만 생성됩니다. 실험마다 새 CoT를 만들고 싶으면 `--no-reuse-cot`를 붙이세요.
실험 3은 같은 스텝을 `AI` 강제 조건과 `Human` 강제 조건에 재사용합니다.

Mario:

```bash
uv run python -m procedural_pipeline.execution.cli --game mario --limit 30
```

Sokoban:

```bash
uv run python -m procedural_pipeline.execution.cli --game sokoban --limit 30
```

주요 옵션(두 게임 공통):

```bash
uv run python -m procedural_pipeline.execution.cli \
    --game mario \
    --num-runs 20 \
    --temperature 1.0 --top-p 1.0 \
    --concurrency 4 \
    --model google/gemini-2.5-pro
```

단일 게임에서 실험 조건을 직접 지정할 수도 있습니다.

```bash
uv run python -m procedural_pipeline.execution.cli --game mario --experiment creator_judgment
uv run python -m procedural_pipeline.execution.cli --game mario --experiment no_creator
uv run python -m procedural_pipeline.execution.cli --game mario --experiment forced_creator --forced-creator AI
uv run python -m procedural_pipeline.execution.cli --game mario --experiment forced_creator --forced-creator Human
```

Sokoban 전용:

```bash
uv run python -m procedural_pipeline.execution.cli --game sokoban --solver-power 500000000
```

## 동작 요약

1. 시작 시 G-Eval 방식으로 **평가 Steps를 1회 생성**해서 콘솔에 출력하고 `evaluation_steps_<timestamp>.json`에 저장
2. 각 맵마다
   - `outputs/video/<game>/<map_id>.mp4`에 영상이 있으면 재사용, 없으면 새로 렌더해서 해당 위치에 저장
   - `temperature=1`, `top_p=1`로 **20회 호출** → 산술 평균/최빈값으로 집계
   - `raw/<map_id>.json`에 20회 원본 + 집계 + 평가 Steps 저장
   - 매 맵 후 `results_<timestamp>.csv`를 갱신 (중간에 끊겨도 안전)
3. 환경 변수: `OPEN_ROUTER_API_KEY` 또는 `OPENROUTER_API_KEY`

영상 캐시 위치를 바꾸고 싶으면 `--video-cache-dir`를 지정하세요. 파일은 항상 `<video-cache-dir>/<game>/<map_id>.mp4` 형태로 저장됩니다.

## 분석

```bash
uv run python -m procedural_pipeline.analyze_results --game mario
uv run python -m procedural_pipeline.analyze_results --game sokoban
uv run python -m procedural_pipeline.analyze_paper_proxy --game mario
uv run python -m procedural_pipeline.analyze_combined
```

분석도 새 경로를 직접 사용할 수 있습니다. 예: `python -m procedural_pipeline.analysis.results --game mario`.
기존 `procedural_pipeline.main`, `procedural_pipeline.analyze_*` 명령은 호환용 wrapper입니다.

- `--results-csv`를 지정하지 않으면 해당 게임의 `output_dir`에서 가장 최근 `results_*.csv`를 자동으로 사용합니다.
- **CSV가 아예 없으면** 해당 게임의 파이프라인을 기본값(`--limit 30 --num-runs 20 --concurrency 4` 등)으로 **자동 실행**해서 CSV를 먼저 만든 뒤 분석합니다.
- 이 자동 실행을 끄고 에러로 막고 싶으면 `--skip-run`을 붙이세요.
- 자동 실행에 쓰는 파이프라인 파라미터는 분석 CLI 옵션(`--limit`, `--num-runs`, `--model`, `--concurrency`, `--solver-power` 등)으로 모두 덮어쓸 수 있습니다.
- `analyze_combined`는 두 게임 중 CSV가 없는 쪽만 자동으로 파이프라인을 돌려 줍니다.
