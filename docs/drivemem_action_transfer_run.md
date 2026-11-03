# Action transfer v2: запуск на сервере

Реализованы диагностика весов/ESS/поддержки действий, F top-16 sidecar labels, B0–B3, F-выбор bandwidth и основного семейства, grouped CV для full-embedding ridge и lookup-оценка development. Это первый цикл enriched-memory экспериментов. Executed-only, utility gate и расширение данных — последующие задачи из плана.

Исходные данные: `/workspace/exp_v2/drivemem_memory/budget20gb_v1`. Новый experiment root: `/workspace/exp_v2/drivemem_memory/experiments/action_transfer_v2`. Его можно переопределить через `DRIVEMEM_ACTION_ROOT`, сохраняя значение во всех командах. Код v1 и его frozen source identity не меняются; новые labels и модели записываются только в новый root. Изменение v2-кода после начала эксперимента требует нового experiment root. В source root ничего не дописывается.

После commit/push и pull на сервере:

```bash
docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh action-check

docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh action-plan
```

`action-plan` не вызывает simulator и не создаёт experiment root. Проверяет исходный frozen identity, M/F labels/features и считает недостающие fit top-16 действия. Максимум 256 новых действий + 32 baseline repeats, то есть 576 raw/filtered scorer calls. GPU и новые изображения не нужны. В `action-check` синтетические tests выполняются в временной папке репозитория, затем проверяется Hydra compose.

Полный запуск:

```bash
docker exec -d -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim bash -c \
  'bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh action-all >> /workspace/exp_v2/drivemem_action_transfer_v2.log 2>&1'

docker exec egorov_drivesuprim tail -n 60 /workspace/exp_v2/drivemem_action_transfer_v2.log
```

Стадии `action-all`:

1. Диагностика F/V по features, без открытия V outcomes.
2. Доразметка F top-16, повтор baseline для parity, сохранение после каждого нового действия. Повтор команды продолжает эти файлы. Уже завершённые действия повторно не оцениваются; baseline repeats остаются.
3. Выбор h по F для четырёх семейств, объявление primary по F. Scale h0 берётся из M с исключением всей query-сессии; три локальные ширины 0.5h0/h0/2h0; beta=.5 и пять доноров фиксированы.
4. No-memory ridge: полный нормированный visual embedding, ego history, geometry, heads, context/action interactions. Регуляризация 1/10/100/1000 выбирается leave-one-F-session-out, при обучении используются M и остальные F-сессии. Нормировки вычисляются только на train части fold; после выбора fit на M+F. Решение в sample space с одной eigendecomposition на fold ограничивает размер матрицы числом labels. Эта стадия может занимать заметное CPU-время; при прерывании незавершённый fit вычисляется заново.
5. Сохранение frozen fit/model hashes и решений всех методов для V до открытия V outcomes.
6. Lookup по существующему development oracle top-16. Дополнительных V simulation calls нет. C/T outcomes не открываются.

По умолчанию launcher ограничивает BLAS/OMP/MKL четырьмя потоками. Изменить через `-e DRIVEMEM_CPU_THREADS=...`. Не запускать несколько writers на один experiment root. GPU не требуется даже при наличии GPU в контейнере.

Отдельные стадии: `action-diagnose`, `action-label`, `action-fit`, `action-evaluate`, `action-report`. Для fit/evaluate необходимы завершённые sidecar labels. `action-evaluate` при необходимости сначала выполняет F-fit, но всегда до чтения V outcomes. Основные диагностические признаки — mean kernel ESS, entropy, расстояние action support, разброс donor residuals и disagreement доноров; все измерены для top-16 query пула, поэтому не обязаны совпасть с прежними агрегатами по 256 кандидатам.

Результат:

```bash
docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh action-report
```

Файлы нового root:

- `identity.json` — source/code/runtime hashes и фиксированный протокол;
- `diagnostics.csv` — диагностика по F/V запросам;
- `fit_labels/<token>.json` — исходные F labels плюс недостающий top-16;
- `fit.json`, `ridge.npz` — результаты выбора только по M/F и frozen контроль;
- `development_decisions.json` — выбор методов до чтения development outcomes;
- `development_per_scene.csv`, `development_by_session.csv`, `report.json` — сравнение качества/компонентов/harm/coverage, paired primary−ridge CI, oracle regret/reserve и три random seed.

B0/B1 — абсолютные residuals с исходным/локальным kernel; B2/B3 — contrast относительно baseline с исходным/локальным kernel. Также доступны scene-only residual и random donor controls для выбранного primary. B0 должен воспроизводить исходную формулу, но ограничение выбора top-16 может изменить его решения относительно опубликованного в чате v1 с пулом 256. Все новые методы выбирают из того же top-16. Значения исходных model scores/imitation сохраняются.

Random результаты выводятся по каждому из трёх заранее заданных seed и агрегируются mean/std; удачный seed не выбирается. Метрики raw safety включают NC/DAC/DDC/TLC/TTC, отдельно сообщаются scalar quality degradation, absolute violations, filtered gains и все покомпонентные Δ в CSV. Средние CI считаются по session blocks. Три V-сессии и повторное использование development не дают подтверждающего теста на популяции. Local PDM не включает two-frame comfort; empirical H не является гарантией безопасности.

Пакет не проводит новые большие oracle, не меняет planner и не расширяет банк. Следующая развилка определяется таблицей B0–B3 vs full-embedding ridge и random controls, а не ростом числа вмешательств.

## После результата: вклад одной сцены и доступность новых сессий

Для аудита крупнейшего вмешательства по существующим raw/filtered outcomes (без переоценки):

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  python -u scripts/drivesuprim/evaluation/audit_drivemem_action_results.py \
  /workspace/exp_v2/drivemem_memory/experiments/action_transfer_v2
```

Показывает долю gain крупнейшей сцены, результаты без неё, все восемь компонентов, multiplicative product, weighted snapshots и проверяет соответствие frozen decisions/model/oracle hashes. PDM 0→1 при неизменных NC/TTC/progress может объясняться снятием другого multiplicative penalty; это необходимо проверить по полному вектору, не называть такой случай исправлением столкновения заранее.

Новые validation-сессии в имеющихся архивах можно посчитать без выбора кадров по качеству и без изменения исходного эксперимента:

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  python -u download/plan_drivemem_validation.py \
  /workspace/exp_v2/drivemem_memory/budget20gb_v1
```

Команда использует исходный `pilot_v1/inventory.csv`, проверяет его frozen hash, исключает все 17 сессий текущего пилота, дополнительные зарезервированные C/T исходного pilot_v1 и official navtest sessions. Другие использованные наборы передавать повторяемым `--exclude-run /path/to/run` (формат splits.json records); по умолчанию неизвестная история внешних экспериментов не учитывается. Метаданные новых логов должны существовать. Для каждой новой сессии нужен один сегмент с минимум четырьмя сценами через 10 секунд. Кадры ещё не фиксируются, новый split не создаётся.

Скачиваются только небольшие официальные JSON manifests; большие архивы не загружаются. Локальные архивы проверяются только по размеру, SHA256 нужно проверить перед извлечением. Вывод содержит новые сессии в четырёх прежних shards, наличие локальных .tgz, shortfall относительно 12 сессий и до пяти вариантов дополнительных архивов, если текущего кэша недостаточно. Это варианты по отдельности, их покрытие может пересекаться. Пути можно переопределить `--inventory-run`/`--archive-dir`. Выборка из тех же архивов сохраняет archive-selection bias; больший размер внутри одной старой сессии не считается новой независимой сессией.
