# 저장소 루트(tutorial)에서 실행:
#   .\scenarios\sokoban\run_sturgeon_seed_match.ps1
# 또는 인자 전달:
#   .\scenarios\sokoban\run_sturgeon_seed_match.ps1 -MaxSeed 200 -Threshold 0.85

param(
    [string]$SturgeonRoot = "sokoban_ref/sturgeon-pub",
    [string]$MapJson = "scenarios/sokoban/map.json",
    [int]$MaxSeed = 500,
    [double]$Threshold = 0.8
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repoRoot

python scenarios/sokoban/sturgeon_seed_match.py `
    --sturgeon-root $SturgeonRoot `
    --map-json $MapJson `
    --max-seed $MaxSeed `
    --threshold $Threshold
