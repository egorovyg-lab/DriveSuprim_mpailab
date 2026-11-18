# Astra: oracle256 на 651 TRAIN-сцене и проверка ограничения Drex

## Существенное исправление плана

В официальном `drex_max/model.py` на зафиксированном commit
`b932571901f531b2378b356fcab7d176048c2bcb` установлено `MAX_OPTIONS = 255`.
Readout — `Linear(hidden, MAX_OPTIONS)`, а `_prompt` отвергает 256 вариантов.
Это ограничение обученной головы, а не количества свободной GPU-памяти.

Прямой запрос с 256 возможными ответами **не поддерживается** этим checkpoint.
Простое увеличение константы не создаст обученный 256-й выход. Нельзя молча
удалить кандидата, смешать вероятности нескольких запросов или объявить
иерархический выбор эквивалентом direct256. Эти варианты потребуют отдельного
протокола. Старый A100 launcher теперь тоже останавливается до загрузки весов.

`astra_drex_probe.py` проверяет исходник по Git blob SHA и скачивает только два
небольших файла с зафиксированных ревизий: model.py и decision_config.json.
Проверяются размер readout, количество кодов/token IDs и формат checkpoint.
Сохраняются сведения о GPU, RAM, cgroup и Python. Проверка не устанавливает
пакеты, не скачивает веса, не выделяет CUDA-память и не читает исходы датасета.
Ожидаемый результат: `blocked_choice_capacity`, exit code **2**. Сетевая или
техническая ошибка имеет статус `technical_failure`, exit code **1**.
Это не успешный inference/quantization smoke test. 4-битная версия пока не
реализована: она не снимает обнаруженное ограничение числа ответов.

Источник ограничения:
https://github.com/nace-ai/drex-decision-models/blob/b932571901f531b2378b356fcab7d176048c2bcb/drex_max/model.py

## Ночной CPU-эксперимент

Все 651 сцена / 32 сессии из ранее исследованного TRAIN, все 256 исходных
кандидатов на сцену: всего **166656** измеренных кандидатов. GPU и Drex не
используются. Две CPU/BLAS-нити, без изменения окружения DS и scorer.

Переиспользуются после проверок:

- исходный полный top64: `rollout_audit_pool64_v1`;
- дополнительные измерения завершённого `controls_oracle64_v1`, созданного
  точной версией `astra_cpu256.py` из commit cd0041e;
- собственные завершённые кандидаты при продолжении этого нового запуска.

Проверяются SHA request/identity/report, совпадение TRAIN-кохорты, пакета,
физических ID, рангов, scorer/cache bindings, receipt каждого кандидата,
диапазон raw PDM и конечность непрерывных метрик. Незавершённый JSON без
receipt повторно измеряется. Повреждённый receipt останавливает расчёт.
На каждой сцене исходный DS повторно симулируется с проверкой raw и
непрерывных метрик; после успешного anchor он фиксируется и при resume
повторно не считается. Каждая новая симуляция загружает свежую копию кеша.

Мини-прогон: четыре разные сессии, две сцены из старой oracle64 и две вне неё.
Так проверяются и reuse, и новый скоринг хвоста. После unit-тестов и мини-прогона
автоматически начинается полный расчёт. Лимита 7 часов нет. При старых полных
результатах ожидается около **112375** новых кандидатов плюс 651 anchor;
порядок длительности — 10–12 часов, не гарантия. Фактическая скорость
публикуется в progress.json. Сборка per_candidate CSV идёт по сценам,
без удержания всех 166 тысяч словарей метрик в памяти.

## Установка на Astra без смены рабочей ветки

Команды ниже выполняются на **хосте Astra**, `egorov@astra`, вне контейнера.
В контейнере нет SSH для `git fetch`: fetch делает хост через его существующие
Git credentials. Bind mount определяется через Docker, файлы checkout не меняются.
Новые инструменты экспортируются через `git show`; старые tools/results не
перезаписываются. Повторный экспорт тех же файлов разрешён, другого содержимого — нет.

```bash
(
set -euo pipefail
HOST_REPO=$(docker inspect egorov_drivesuprim --format '{{json .Mounts}}' |
  python3 -c 'import json,sys; from pathlib import PurePosixPath as P; target=P("/workspace/DriveSuprim"); mounts=[m for m in json.load(sys.stdin) if m["Type"]=="bind" and (target==P(m["Destination"]) or P(m["Destination"]) in target.parents)]; m=max(mounts,key=lambda x:len(x["Destination"])); print(P(m["Source"])/target.relative_to(P(m["Destination"])))')
git -C "$HOST_REPO" fetch origin codex/drex-max

docker exec -i -w /workspace/DriveSuprim egorov_drivesuprim bash -s <<'BASH'
set -euo pipefail
ROOT=/workspace/exp_v2/drex_astra_cpu256/oracle651_v2
TOOLS=$ROOT/tools
COMMIT=$(git rev-parse origin/codex/drex-max)
mkdir -p "$TOOLS"
for f in astra_cpu256.py run_astra_cpu256.sh test_astra_cpu256.py astra_drex_probe.py run_astra_night.sh test_astra_night.py; do
  git show "$COMMIT:scripts/drex/$f" > "$TOOLS/$f.pending"
  if [[ -e "$TOOLS/$f" ]]; then
    cmp -s "$TOOLS/$f" "$TOOLS/$f.pending" || { echo "Tools differ: $f; use a new output directory" >&2; exit 1; }
    rm -- "$TOOLS/$f.pending"
  else
    mv -- "$TOOLS/$f.pending" "$TOOLS/$f"
  fi
done
export ASTRA_CPU256_ROOT=$ROOT
bash "$TOOLS/run_astra_night.sh" start
BASH
)
```

После `launched` терминал можно закрыть: дочерний процесс отвязан от сессии
и stdin/stdout/stderr. Повторный `start` продолжает тот же запуск; file lock
предотвращает одновременное исполнение. Если вернулся `Already running`,
дополнительный процесс не создан. После перезапуска контейнера нужен `start`.

Проверка контракта Drex запускается отдельно и не отменяет ночной CPU-расчёт:

```bash
docker exec egorov_drivesuprim bash \
  /workspace/exp_v2/drex_astra_cpu256/oracle651_v2/tools/run_astra_night.sh probe
```

Статус:

```bash
docker exec egorov_drivesuprim bash \
  /workspace/exp_v2/drex_astra_cpu256/oracle651_v2/tools/run_astra_night.sh status
```

`pipeline_state=running` означает, что lock удерживается живым процессом;
`idle` само по себе не означает успех. Завершение подтверждается
`report.status=complete`, `scenes=651`, `oracle_scenes=651`,
`completed_candidates=166656`, `anchors_parity_passed=651`.
При ошибке смотреть failure.json и run.log. Не удалять identity/checkpoints.
После устранения внешней причины повторить тот же `start` теми же tools.

## Результаты

В `/workspace/exp_v2/drex_astra_cpu256/oracle651_v2`:

- `report.json`: DS, два простых контроля, PDM/relative-safe/comfort oracle16/64/256;
- `oracle_pool_expansion` в отчёте: парные прибавки 64−16, 256−64, 256−16,
  session-macro bootstrap, количество сцен/сессий с прибавкой и доля двух
  крупнейших сцен в положительной micro-прибавке;
- `per_scene.csv.gz`, `per_candidate.csv.gz`, request.json, identity.json;
- `progress.json`, `run.log`, покандидатные checkpoints в scenes/;
- `drex_probe/report.json`: отдельный технический результат, не оценка вождения.

Oracle использует известные исходы и остаётся верхней границей. Критерии
комфорта и geometry_peaks/progress budget не смягчаются для получения замен.
Набор не стал независимым или более разнообразным: по-прежнему 651 сцена и
32 сессии. Holdout не открывается. Метрики — raw local PDM, не полный EPDMS.

Проверки локально/в контейнере с доступными исходными helpers:
`bash .../tools/run_astra_night.sh tests`. Они используют синтетические
fixtures и не запускают ни Drex, ни настоящий scorer.
