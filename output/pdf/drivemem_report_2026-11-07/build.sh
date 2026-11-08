#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for pass in 1 2; do
  xelatex -interaction=nonstopmode -halt-on-error report.tex
done
