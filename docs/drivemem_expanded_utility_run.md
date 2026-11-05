# Контроль эффекта новых M/F-данных

5 ноября 2026. Сбор shard174 завершён: 32 label-сцены, без пропусков.
Новые ухудшения: M — DAC16, NC2, TTC2; F — NC5, TTC8. DDC и относительные
TLC-ухудшения отсутствуют; положительных изменений пяти safety-компонентов
тоже нет. Количества относятся к коррелированным действиям, не к независимым
эпизодам. В F новые NC/TTC-ухудшения приходятся на одну сессию.

Следующий контроль сохраняет прежний pair-utility алгоритм, признаки, target,
нулевой порог, lambda grid и процедуру macro F-CV. Добавляются только готовые
M/F-сцены. h и lambda переоцениваются прежней процедурой, не фиксируются численно.
Переобучаются direct_memory и direct_no_memory. Оба получают одинаковые labels.

Данные после объединения: M64/8 сессий, F48/6 сессий. Используются полные top16
старых и новых сцен. Старый случайный хвост не добавляется, как и в pair_utility_v1.
CV macro objective включает все F-сцены: так сохраняется процедура старого
алгоритма. Новый F содержит uniform и challenge; это явно выбранный data-control,
а не оценка качества при естественном распределении и не риск-калибровка.

Источник старых меток определяется замороженным `pair_utility_v1/identity.json`:
M top16 — memory_top16_v1, F top16 — action_transfer_v2. Новый source:
`mf_new4_shard174_v1`. Старые roots читаются без изменений.

Нет GPU, image transfer, simulator/scorer calls или новых labels.
Групповые folds F исключают целые сессии; M-запрос не получает доноров своей
сессии. Все решения обоих прежних development-наборов фиксируются до открытия
их outcomes. C/T labels не читаются.

## Запуск

После переноса изменений через Git:

```bash
dm_refit() {
  docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
    bash scripts/drivesuprim/evaluation/run_drivemem_expanded_utility.sh "$@"
}

dm_refit check && dm_refit plan
dm_refit all
```

Plan проверяет provenance и показывает counts без обучения и создания output.
All выполняет fit, фиксацию решений и lookup готовых query outcomes. CPU BLAS
ограничен четырьмя потоками. Доступны также fit, evaluate, report. Повторный all
проверяет model hashes и использует уже зафиксированные модели.

Output:
`/workspace/exp_v2/drivemem_memory/experiments/pair_utility_expanded_v1`.

В нём identity.json, fit.json, direct_memory.npz, direct_no_memory.npz,
query_decisions.json, report.json, per_scene/by_session CSV двух query-когорт.

```bash
docker exec egorov_drivesuprim cat \
  /workspace/exp_v2/drivemem_memory/experiments/pair_utility_expanded_v1/report.json
```

Сравнивать абсолютные gain/harm/coverage с baseline, парную разность memory
против no-memory и `comparison_to_original`: изменение результатов относительно
pair_utility_v1 для каждого метода. Не считать успехом только то, что память
менее вредна, чем контроль; нужен выигрыш над DriveSuprim и устойчивость по сессиям.

Оба query-набора уже просмотрены. Их повторная оценка остаётся development,
даже если модель обучалась на новых сессиях. Отчёт не доказывает пользу retrieval,
нулевой risk, continual improvement или full EPDMS gain. После этого data-control
решаем, оправдана ли новая ограниченная компонентная семья. Новые загрузки и
калибровка риска до этого решения не запускаются.
