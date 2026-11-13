# DriveMem residual memory top64: запуск на Astra

Реализует [зафиксированный план](drivemem_residual_memory_plan_2026_11_13.md).
651 training-сцена / 32 сессии, одинаковый top64 и supervision для всех
вариантов. Четыре session folds, 10 base fits + 4 residual-MLP fits.
Свежие holdout и прежние development-когорты не оцениваются.

Это CPU-эксперимент для `egorov_drivesuprim` на Astra. Две GTX 1080 остаются
свободны; A100 не требуется. Используются существующие NumPy/Pandas/PyTorch.
Нет новых загрузок изображений, DS export, Drex-запросов или scorer calls.

## Обновление и подготовка

В host-каталоге репозитория на Astra, в ветке `main`:

```bash
git pull --ff-only origin main

dm_rm() {
  docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
    bash scripts/drivesuprim/evaluation/run_drivemem_residual_memory.sh "$@"
}

dm_rm check && dm_rm smoke && dm_rm inventory
```

`check` проверяет исключение сессий, соответствие меток и физических IDs,
совпадение признаков со старым селектором, восстановление checkpoint и
блокировку отчёта до завершения OOF. `smoke` выполняет все 14 коротких fits
на искусственных данных, затем проверяет возобновление. Временный smoke
каталог удаляется; реальные labels и holdout для smoke не открываются.

В успешном `inventory` ожидается:

```json
{"status": "ready", "scenes": 651, "sessions": 32, "topk": 64,
 "candidate_labels": 41664, "expected_fits": 14, "device": "cpu"}
```

По умолчанию читается bundle:
`/workspace/exp_v2/drivemem_memory/selector_training_bundle_v3_30gb`.
Top64 labels ищутся только в
`/workspace/exp_v2/drex_rollout_audit/rollout_audit_table_*/artifacts/labels.json`.
Проверяются report, identity, исходный audit request и portable package
manifest; для общих top16 обязательна parity raw-меток. Scalar PDM не
реконструируется из компонентов.

Если auto-discovery остановился, скопируй его ошибку: она перечисляет
проверенные пути и причины. Для явного указания существующих файлов:

```bash
dm_rm inventory \
  --labels /путь/к/artifacts/labels.json \
  --package-manifest /путь/к/package_manifest.json
```

Затем передай те же параметры команде `plan`. Для перенесённых меток рядом
нужны их `report.json`, `identity.json` и родительский `audit_request.json`;
последний можно указать через `--label-request`. Это только числовые
артефакты прежнего измерения. При несовпадении пакетов, неполных labels или
неверном числе сессий запуск прекращается, метрики не пересчитываются.

После успешной инвентаризации:

```bash
dm_rm plan && dm_rm pilot
```

`plan` фиксирует protocol, четыре группы сессий, hashes и отдельную копию
input/label данных. Старый bundle не изменяется. `pilot` обучает одну
реальную base-модель на двух training groups, показывает время и peak RSS
процесса на Linux. Query outcomes для оценки не читает. Готовый fit входит
в бюджет 14 и повторно используется командой `all`.

## Основной расчёт и возобновление

После успешных предыдущих этапов:

```bash
docker exec -d -w /workspace/DriveSuprim egorov_drivesuprim \
  bash -c 'exec bash scripts/drivesuprim/evaluation/run_drivemem_residual_memory.sh all > /workspace/exp_v2/drivemem_residual_memory_top64_v1.log 2>&1'
```

Процесс работает после отключения терминала. Контроль:

```bash
docker exec egorov_drivesuprim \
  tail -n 60 /workspace/exp_v2/drivemem_residual_memory_top64_v1.log

dm_rm status
```

После аварийной остановки повтори ту же команду `all`. Готовые fits и folds
проверяются и используются повторно. Незавершённый fit обучается заново с
фиксированным seed. OS-lock блокирует второй writer; `.run.lock` удалять
не нужно. Прежние ошибки сохраняются в `failures/`, а `status.last_failure`
показывает последнюю историческую ошибку даже после успешного продолжения.

По умолчанию два CPU threads, модели float32, retrieval/отчёт float64.
Поиск блоками, без полного query×donor×feature tensor. Полное время заранее
не гарантируется: кроме 14 fits есть поиск доноров и 10 000 bootstrap draws.

## Результат

```bash
dm_rm report

docker exec egorov_drivesuprim cat \
  /workspace/exp_v2/drivemem_memory/experiments/residual_memory_top64_v1/summary.json
```

Основной root:
`/workspace/exp_v2/drivemem_memory/experiments/residual_memory_top64_v1`.

- `identity.json`, `split.json`, `data/manifest.json`: условия и происхождение.
- `fits/*`: checkpoint, история/время/peak RSS, hashes и replay каждого fit.
- `folds/*`: источники cross-fitted остатков, разные donor sessions,
  перестановки, расстояния/веса, предсказания и решения всех вариантов.
- `per_scene.csv`: фактический PDM, потери, missed repairs, ранг, восемь
  компонентов и прямые изменения против выбора C1.
- `per_session.csv`: эффект и ухудшения по каждой сессии/методу.
- `paired_comparisons.json`, `bootstrap_draws.npz`, `summary.json`: парные
  сравнения, устойчивость по сессиям и заранее заданные критерии продолжения.

Методы: `C0_DS`, `C1_selector`, `C2_memory`, `C3_corrector`, пять фиксированных
`C4_*` random и пять `C5_*` shuffled. `C4_seed_mean`/`C5_seed_mean` в summary —
средний эффект пяти policies, не дополнительные исполняемые policies.

Основной вопрос — C2 против C1, C3 и отрицательных контролей. Успех должен
пройти все поля `criteria`, включая минимум +0,25 п.п. сверх C1, положительные
парные интервалы, устойчивость без лучших сессий и отсутствие новых
safety-degraded сцен там, где C1 их избегал. `primary=DS` сохраняется:
`continue_memory_hypothesis=true` означает повод для независимой проверки,
а не автоматическое внедрение или подтверждённую безопасность.

Изменение кода, input hashes, runtime, числа threads или настроек требует
нового root через `--out`. Идентичности/commit-файлы не удалять для обхода
проверок. При одинаковых условиях повтор завершённого отчёта проверяет
его hashes и читает результат без нового обучения или оценки.

Launcher также поддерживает `night` (check → smoke → plan → pilot → all).
Переменные окружения: `DRIVEMEM_RESIDUAL_ROOT`, `DRIVEMEM_RESIDUAL_BUNDLE`,
`DRIVEMEM_RESIDUAL_LABELS`, `DRIVEMEM_RESIDUAL_REQUEST`,
`DRIVEMEM_RESIDUAL_PACKAGE_MANIFEST`, `DRIVEMEM_RESIDUAL_THREADS`.
При использовании через Docker передавай их с `docker exec -e ...`;
переменные host-shell сами внутрь контейнера не попадают.
