Param(
    [switch]$SkipDeps,
    [switch]$CheckNeo4j,
    [switch]$Gpu
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python not found in PATH."
}

$venvPath = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[preflight] Creating virtual environment..."
    & python -m venv .venv
}

if (-not $SkipDeps) {
    $requirementsFile = "requirements-cpu.txt"
    if ($Gpu) {
        $requirementsFile = "requirements-gpu.txt"
    }

    Write-Host "[preflight] Installing dependencies from $requirementsFile..."
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r $requirementsFile
    & $venvPython -m pip install -e . --no-deps
}

Write-Host "[preflight] Running syntax compilation check..."
& $venvPython -m compileall src scripts

$smokeArgs = @("scripts/smoke_check.py")
if ($CheckNeo4j) {
    $smokeArgs += "--check-neo4j"
}

Write-Host "[preflight] Running smoke check..."
& $venvPython $smokeArgs

Write-Host "[preflight] Verifying CLI entrypoint..."
& $venvPython -m graphrag.cli --help | Out-Null

Write-Host "[preflight] PASSED"
