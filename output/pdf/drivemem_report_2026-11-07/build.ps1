$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    foreach ($pass in 1..2) {
        & xelatex -interaction=nonstopmode -halt-on-error report.tex
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX failed: $LASTEXITCODE" }
    }
} finally { Pop-Location }
