# Procedural Pipeline Commands

자주 쓰는 실행 커맨드를 모아둔 문서입니다.  
프로젝트 루트(`tutorial`)에서 실행하세요.

## 0) 기본 준비

```bash
pip install -e .
```

OpenRouter 키가 필요합니다.

```bash
# PowerShell
$env:OPENROUTER_API_KEY="YOUR_KEY"
```

## 1) 파이프라인 실행 (평가 + CSV 생성)

### Mario

```bash
python -m procedural_pipeline.main --game mario
```

### Sokoban

```bash
python -m procedural_pipeline.main --game sokoban
```

### 자주 쓰는 옵션 예시

```bash
python -m procedural_pipeline.main --game mario --num-runs 20 --temperature 1 --top-p 1 --concurrency 4 --limit 30
python -m procedural_pipeline.main --game sokoban --num-runs 20 --temperature 1 --top-p 1 --concurrency 4 --limit 30
```

## 2) 결과 분석 (집계 CSV 기준)

### Mario

```bash
python -m procedural_pipeline.analyze_results --game mario
```

### Sokoban

```bash
python -m procedural_pipeline.analyze_results --game sokoban
```

### 결과 CSV 직접 지정

```bash
python -m procedural_pipeline.analyze_results --game mario --results-csv outputs/procedural_mario/results_20260422-214139.csv
```

## 3) Paper Proxy 분석

### Mario

```bash
python -m procedural_pipeline.analyze_paper_proxy --game mario
```

### Sokoban

```bash
python -m procedural_pipeline.analyze_paper_proxy --game sokoban
```

## 4) Combined 분석 (게임 전체)

```bash
python -m procedural_pipeline.analyze_combined
```

## 5) Raw 20-run 개별 분석

### Mario

```bash
python -m procedural_pipeline.analyze_raw_runs --game mario
```

### Sokoban

```bash
python -m procedural_pipeline.analyze_raw_runs --game sokoban
```

### Raw 디렉토리 직접 지정

```bash
python -m procedural_pipeline.analyze_raw_runs --game mario --raw-dir outputs/procedural_mario/raw
```

## 6) 출력 파일 위치

- Mario 평가 결과: `outputs/procedural_mario/`
- Sokoban 평가 결과: `outputs/procedural_sokoban/`
- 일반 분석 출력: `outputs/procedural_analysis/`
- Raw-run 분석 출력: `outputs/procedural_analysis_runs/`

## 7) 자주 보는 파일

- 최신 결과 CSV: `outputs/procedural_mario/results_*.csv`
- 맵별 raw 로그: `outputs/procedural_mario/raw/*.json`
- 평가 스텝: `outputs/procedural_mario/evaluation_steps_*.json`
