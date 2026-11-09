# Сравнение 399 и 651 сцен на одинаковых OOF-примерах

Общие показатели OOF на 399 и 651 сцене смешивают изменение обучающей выборки
с изменением проверочной. Этот этап сравнивает готовые предсказания двух
экспериментов на одних и тех же прежних 399 сценах из 22 поездок. Сборщик не
обучает модели, не вызывает scorer, не скачивает изображения и не открывает
старые development или fresh holdout. Требуются только NumPy и pandas, CPU.

После commit/push/pull внутри контейнера:

```bash
cd /workspace/DriveSuprim
bash scripts/drivesuprim/evaluation/run_selector_matched.sh check
bash scripts/drivesuprim/evaluation/run_selector_matched.sh compare \
  > /workspace/exp_v2/selector_matched_399_vs_651.json
cat /workspace/exp_v2/selector_matched_399_vs_651.json
```

Входы: `experiments/trajectory_selector_v2` и
`experiments/trajectory_selector_v3_30gb_e120`, bundles
`selector_training_bundle_v2` и `selector_training_bundle_v3_30gb` внутри
`/workspace/exp_v2/drivemem_memory`. CLI flags `--reference`, `--expanded`,
`--old-bundle`, `--new-bundle` позволяют явно задать другие пути.

Проверяются hashes готовых fit-артефактов и CV folds, совпадение model code,
config, epoch count, optimizer settings и runtime. Для каждого из 399 общих
примеров проверяются metadata, dtype и значения всех input/label arrays в
обоих bundles. Сверяются baseline/oracle scores и полное покрытие CSV без
дубликатов. Для каждого fold проверяется исключение всей поездки, включая
19 новых сцен в трёх прежних поездках. Несовместимые условия вызывают ошибку,
а не автоматическое ослабление сравнения. Каталоги экспериментов не изменяются.

В отчёте:

- `shared.reference` и `shared.expanded`: обе модели на одинаковых старых сценах.
- `shared.paired`: разность expanded minus reference, с равным весом поездок,
  числом поездок с улучшением/ухудшением и парным session bootstrap.
- `selected_gates`: разные настроенные пороги в двух экспериментах.
- `added_scenes`: все 252 дополнительных примера новой модели отдельно.
- `added_new_session_scenes`: 233 сцены из десяти новых поездок.
- `added_old_session_scenes`: 19 дополнительных сцен трёх прежних поездок.
- `worst_expanded_raw_decisions`: несколько худших решений с прогнозами и
  изменениями компонентов, отдельно для общих и дополнительных сцен.

Для анализа эффекта обучения на большем наборе сначала смотрим raw policies
на общих сценах: правила выбора, архитектура и настройки фиксированы. В gated
policies поменялись ещё и подобранные thresholds, поэтому это оценка всей
настроенной системы, а не изолированный эффект данных. Пороги и эпохи уже
настраивались по OOF; такие результаты не являются независимым подтверждением.
Bootstrap описательный: CV folds используют пересекающиеся training sets.
Local PDM не равен EPDMS, нулевой observed harm не является гарантией безопасности.
