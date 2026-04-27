# Procedural Pipeline (Unified)

Mario / Sokoban 두 게임을 한 폴더에서 공통 파이프라인으로 평가합니다.
공통 로직(G-Eval Auto-CoT, 20회 평균, CSV/Raw 저장, 분석 스크립트)은 모두 `procedural_pipeline/`에 있고, 게임별 차이(맵 파싱, 렌더러, 기본 경로)만 `procedural_pipeline/games/*.py`로 분리되어 있습니다.

## 폴더 구조

```
procedural_pipeline/
    judge.py                  # 공통 - G-Eval CoT, 멀티런, 프롬프트, 게임 프로필
    csv_output.py             # 공통 - 결과 CSV
    runner.py                 # 공통 - 파이프라인 실행 루프
    main.py                   # 공통 CLI (--game mario|sokoban)
    analyze_results.py        # 공통 분석 (--game ...)
    analyze_paper_proxy.py    # 공통 논문 비교 (--game ...)
    analyze_combined.py       # 두 게임 합친 논문 비교
    games/
        mario.py              # 맵 정규화 + Mario jar 렌더
        sokoban.py            # 맵 정규화 + pcg_benchmark 렌더
```

## 실행

Mario:

```bash
uv run python -m procedural_pipeline.main --game mario --limit 30
```

Sokoban:

```bash
uv run python -m procedural_pipeline.main --game sokoban --limit 30
```

주요 옵션(두 게임 공통):

```bash
uv run python -m procedural_pipeline.main \
    --game mario \
    --num-runs 20 \
    --temperature 1.0 --top-p 1.0 \
    --concurrency 4 \
    --model google/gemini-2.5-pro
```

Sokoban 전용:

```bash
uv run python -m procedural_pipeline.main --game sokoban --solver-power 500000000
```

## 동작 요약

1. 시작 시 G-Eval 방식으로 **평가 Steps를 1회 생성**해서 콘솔에 출력하고 `evaluation_steps_<timestamp>.json`에 저장
2. 각 맵마다
   - 기존 `<map_id>.mp4`가 있으면 재사용, 없으면 새로 렌더
   - `temperature=1`, `top_p=1`로 **20회 호출** → 산술 평균/최빈값으로 집계
   - `raw/<map_id>.json`에 20회 원본 + 집계 + 평가 Steps 저장
   - 매 맵 후 `results_<timestamp>.csv`를 갱신 (중간에 끊겨도 안전)
3. 환경 변수: `OPEN_ROUTER_API_KEY` 또는 `OPENROUTER_API_KEY`

## 분석

```bash
uv run python -m procedural_pipeline.analyze_results --game mario
uv run python -m procedural_pipeline.analyze_results --game sokoban
uv run python -m procedural_pipeline.analyze_paper_proxy --game mario
uv run python -m procedural_pipeline.analyze_combined
```

- `--results-csv`를 지정하지 않으면 해당 게임의 `output_dir`에서 가장 최근 `results_*.csv`를 자동으로 사용합니다.
- **CSV가 아예 없으면** 해당 게임의 파이프라인을 기본값(`--limit 30 --num-runs 20 --concurrency 4` 등)으로 **자동 실행**해서 CSV를 먼저 만든 뒤 분석합니다.
- 이 자동 실행을 끄고 에러로 막고 싶으면 `--skip-run`을 붙이세요.
- 자동 실행에 쓰는 파이프라인 파라미터는 분석 CLI 옵션(`--limit`, `--num-runs`, `--model`, `--concurrency`, `--solver-power` 등)으로 모두 덮어쓸 수 있습니다.
- `analyze_combined`는 두 게임 중 CSV가 없는 쪽만 자동으로 파이프라인을 돌려 줍니다.
