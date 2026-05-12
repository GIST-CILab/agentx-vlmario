# Procedural Pipeline Commands

자주 쓰는 실행 커맨드를 모아둔 문서입니다.  
프로젝트 루트(`tutorial`)에서 실행하세요. **의존성 실행은 `uv` 기준**입니다.

## 0) 기본 준비

```bash
uv sync
```

프로젝트를 editable로만 맞추고 싶다면(락 없이):

```bash
uv pip install -e .
```

OpenRouter 키가 필요합니다.

```bash
# PowerShell
$env:OPENROUTER_API_KEY="YOUR_KEY"
```

이하 모든 `python -m ...` 호출은 **`uv run`**으로 같은 환경에서 실행됩니다.

## 1) 파이프라인 실행 (평가 + CSV 생성)

### Mario

```bash
uv run python -m procedural_pipeline.main --game mario
```

### Sokoban

```bash
uv run python -m procedural_pipeline.main --game sokoban
```

### 자주 쓰는 옵션 예시

```bash
uv run python -m procedural_pipeline.main --game mario --num-runs 20 --temperature 1 --top-p 1 --concurrency 4 --limit 30
uv run python -m procedural_pipeline.main --game sokoban --num-runs 20 --temperature 1 --top-p 1 --concurrency 4 --limit 30
```

## 2) 결과 분석 (집계 CSV 기준)

### Mario

```bash
uv run python -m procedural_pipeline.analyze_results --game mario
```

### Sokoban

```bash
uv run python -m procedural_pipeline.analyze_results --game sokoban
```

### 결과 CSV 직접 지정

```bash
uv run python -m procedural_pipeline.analyze_results --game mario --results-csv outputs/procedural_mario/results_20260422-214139.csv
```

## 3) Paper Proxy 분석

### Mario

```bash
uv run python -m procedural_pipeline.analyze_paper_proxy --game mario
```

### Sokoban

```bash
uv run python -m procedural_pipeline.analyze_paper_proxy --game sokoban
```

## 4) Combined 분석 (게임 전체)

```bash
uv run python -m procedural_pipeline.analyze_combined
```

## 5) Raw 20-run 개별 분석

### Mario

```bash
uv run python -m procedural_pipeline.analyze_raw_runs --game mario
```

### Sokoban

```bash
uv run python -m procedural_pipeline.analyze_raw_runs --game sokoban
```

### Raw 디렉토리 직접 지정

```bash
uv run python -m procedural_pipeline.analyze_raw_runs --game mario --raw-dir outputs/procedural_mario/raw
```

## 6) Ablation: v2 블라인드 평가 + 분석

`v1.3` 결과 + 비디오 + 평가 스텝을 **재사용**하면서, 프롬프트에서 creator_belief / confidence / reasoning을 모두 제거한 평가(v2)를 수행합니다.

### Mario v2 (블라인드)

```bash
uv run python -m procedural_pipeline.main --game mario --output-dir outputs/procedural_mario_v2_blind --blind --eval-steps-file results/v1.3/procedural_mario/evaluation_steps_20260422-214139.json --video-source-dir results/v1.3/procedural_mario
```

### Sokoban v2 (블라인드)

```bash
uv run python -m procedural_pipeline.main --game sokoban --output-dir outputs/procedural_sokoban_v2_blind --blind --eval-steps-file results/v1.3/procedural_sokoban/evaluation_steps_20260423-155941.json --video-source-dir results/v1.3/procedural_sokoban
```

### Ablation 분석 (v1 belief 기준 v2 점수 재집계)

```bash
uv run python -m procedural_pipeline.analyze_blind_ablation --game mario
uv run python -m procedural_pipeline.analyze_blind_ablation --game sokoban
```

### 논문 Table 1 + v1 + v2 한 표 비교

논문의 **인지된 제작자**(Perceived Human / AI)별 Mean(SD)과, 동일 지표로 **v1(믿음별)**·**v2(블라인드 점수를 v1 믿음 구간에 재할당)**를 한 줄에 맞춥니다.

**논문은 Mario+Sokoban을 합친 표**이므로, 우리도 합산하려면 `--combined`를 쓰세요.

```bash
# 게임별 (한 종류만)
uv run python -m procedural_pipeline.analyze_paper_v1_v2_table --game mario
uv run python -m procedural_pipeline.analyze_paper_v1_v2_table --game sokoban

# Mario + Sokoban 풀링 (논문 Table 1과 동일 프레이밍)
uv run python -m procedural_pipeline.analyze_paper_v1_v2_table --combined
```

CSV 직접 지정 (단일 게임):

```bash
uv run python -m procedural_pipeline.analyze_paper_v1_v2_table --game mario `
  --v1-csv results/v1.3/procedural_mario/results_20260422-214139.csv `
  --v2-csv outputs/procedural_mario_v2_blind/results_20260428-151808.csv
```

출력: 단일 게임은 `outputs/procedural_paper_v1_v2_table/` , `--combined`는 `outputs/procedural_paper_v1_v2_table_combined/` 에 `paper_v1_v2_table.md`, `.json`, **`paper_v1_v2_compare.png`** (그래프 생략: `--no-plot`).

### Ablation CSV 직접 지정

```bash
uv run python -m procedural_pipeline.analyze_blind_ablation --game mario `
  --v1-csv results/v1.3/procedural_mario/results_20260422-214139.csv `
  --v2-csv outputs/procedural_mario_v2_blind/results_<latest>.csv
```

## 7) 출력 파일 위치

- Mario 평가 결과: `outputs/procedural_mario/`
- Sokoban 평가 결과: `outputs/procedural_sokoban/`
- Mario v2 (블라인드): `outputs/procedural_mario_v2_blind/`
- Sokoban v2 (블라인드): `outputs/procedural_sokoban_v2_blind/`
- 일반 분석 출력: `outputs/procedural_analysis/`
- Raw-run 분석 출력: `outputs/procedural_analysis_runs/`
- Blind ablation 분석: `outputs/procedural_blind_ablation/`
- Paper+v1+v2 비교 표 (단일 게임): `outputs/procedural_paper_v1_v2_table/`
- Paper+v1+v2 비교 표 (**Mario+Sokoban 풀링**): `outputs/procedural_paper_v1_v2_table_combined/`

## 8) 자주 보는 파일

- 최신 결과 CSV: `outputs/procedural_mario/results_*.csv`
- 맵별 raw 로그: `outputs/procedural_mario/raw/*.json`
- 평가 스텝: `outputs/procedural_mario/evaluation_steps_*.json`
