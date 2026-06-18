## Perceived-source bins (v1 belief; v2 blind scores in same bins)

| Metric | Paper H | Paper AI | v1 H | v1 AI | v1 p | v2 H | v2 AI | v2 p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Enjoyment / Fun | 3.72 (1.20) | 2.92 (1.33) | 3.11 (0.56) | 2.06 (0.51) | <.001 | 2.87 (0.63) | 2.18 (0.57) | <.001 |
| Difficulty / Challenge | 3.65 (1.32) | 3.88 (1.27) | 1.90 (0.55) | 1.52 (0.49) | 0.011 | 2.07 (0.54) | 2.40 (1.08) | 0.538 |
| Frustration / Negative affect | 2.84 (1.43) | 3.60 (1.37) | 1.23 (0.19) | 1.67 (0.83) | 0.098 | 1.41 (0.67) | 2.46 (1.36) | 0.003 |
| Novelty / Surprise | 2.62 (1.26) | 2.65 (1.35) | 1.51 (0.34) | 1.23 (0.24) | 0.001 | 1.69 (0.37) | 1.84 (0.49) | 0.254 |
| Aesthetics / Design quality | 3.57 (1.17) | 2.70 (1.23) | 3.08 (0.47) | 1.89 (0.58) | <.001 | 2.90 (0.62) | 2.11 (0.78) | <.001 |

### Mann–Whitney p (v1 / v2)

**Pooled (Mario + Sokoban)**: Mann–Whitney는 두 게임 맵을 한 풀에 넣어 Human-bin vs AI-bin을 비교합니다 (논문 Table 1과 동일한 ‘합산’ 프레이밍). 다만 두 게임 난이도·스케일이 달라 **교환 가능한 표본**은 아님.

- **v1 p**: full-prompt 맵 점수, Human-believed vs AI-believed 구간.
- **v2 p**: blind 맵 점수, **동일 구간**(구간 = 해당 맵의 v1 `creator_belief`).
- **의미**: 두 구간 분포가 우연히 이 정도로 갈릴 확률. 맵 수·불균형·구간 정의 등으로 **탐색적 지표**로만 해석하는 것이 좋다.


## Ground-truth bins (maps.json: first 15 Human, rest AI)

| Metric | v1 Human (truth) | v1 AI (truth) | v1 p | v2 Human (truth) | v2 AI (truth) | v2 p |
| --- | --- | --- | --- | --- | --- | --- |
| Enjoyment / Fun | 2.41 (0.76) | 2.52 (0.71) | 0.668 | 2.48 (0.69) | 2.41 (0.67) | 0.690 |
| Difficulty / Challenge | 1.61 (0.55) | 1.72 (0.53) | 0.321 | 2.13 (0.75) | 2.43 (1.04) | 0.371 |
| Frustration / Negative affect | 1.52 (0.82) | 1.49 (0.55) | 0.542 | 1.88 (1.14) | 2.24 (1.33) | 0.778 |
| Novelty / Surprise | 1.35 (0.33) | 1.33 (0.29) | 0.994 | 1.74 (0.41) | 1.82 (0.49) | 0.594 |
| Aesthetics / Design quality | 2.32 (0.90) | 2.38 (0.67) | 0.842 | 2.44 (0.84) | 2.38 (0.80) | 0.767 |

### Mann–Whitney p (truth bins, v1 / v2)

**Pooled (Mario + Sokoban)**: Mann–Whitney는 두 게임 맵을 한 풀에 넣어 **진실 Human vs AI** 구간을 비교합니다. 논문 Table 1의 ‘인지된 출처’와는 다른 프레이밍입니다.

- **v1 p**: full-prompt 점수, Human-made vs AI-generated 맵(동일 `maps.json` 순서 규칙).
- **v2 p**: blind 점수, 동일 진실 구간.

