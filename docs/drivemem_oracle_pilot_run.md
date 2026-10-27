# DriveMem: первый тест экспорта и оракула

Скрипт проводит пилот на NAVSIM v2: экспортирует финальные кандидаты DriveSuprim и последовательно оценивает их стандартным `pdm_score`. Он рассчитывает локальный оракул без two-frame extended comfort. Полный EPDMS-оракул и замкнутый контур здесь ещё не реализованы. Для следующего этапа на 32 логах см. `docs/drivemem_multilog_run.md`. Актуальная версия также требует `navsim/evaluate/drivemem_oracle_utils.py`.

## Перед запуском

Код должен находиться в `/workspace/DriveSuprim` внутри контейнера `egorov_drivesuprim`. Помимо нового Python-скрипта и launcher нужны изменения в `drivesuprim_config.py` и `drivesuprim_model.py`. Убедитесь, что актуальные файлы перенесены на сервер; если синхронизируете через git, включите все четыре файла.

Пути по умолчанию: checkpoint `/workspace/exp_v2/model_ckpt/drivesuprim_r34.ckpt`, кэш `/workspace/exp_v2/metric_cache/test/ori`, результаты `/workspace/exp_v2/drivemem_oracle/pilot`. Переменные `NAVSIM_DEVKIT_ROOT`, `NAVSIM_EXP_ROOT`, `OPENSCENE_DATA_ROOT` должны быть заданы в контейнере, как при воспроизведении.

По умолчанию используются первые 8 сцен `navtest_debug`, отобранные SceneLoader. Список токенов фиксируется в manifest: это техническая выборка, а не репрезентативная оценка. При повторении на другой машине порядок файлов и состав первых сцен могут отличаться.

## 1. Экспорт кандидатов на GPU

Из оболочки сервера:

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_oracle_pilot.sh export
```

Экспорт выполняет два прохода модели по каждой сцене с выключенным и включённым сбором кандидатов. Скрипт останавливается, если выбранные траектории различаются, выбранный индекс не соответствует score или словарю, или отсутствует metric cache. Для каждой сцены сохраняются все 256 кандидатов и финальные scores.

Папка результата должна быть новой или пустой. Чтобы выбрать другую:

```bash
docker exec -e DRIVEMEM_OUTPUT=/workspace/exp_v2/drivemem_oracle/pilot2 \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_oracle_pilot.sh export
```

## 2. Оценка 16 кандидатов на сцену

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_oracle_pilot.sh score
```

Для каждой сцены отбираются 16 кандидатов с наивысшим model score, включая baseline. Каждый кандидат оценивается отдельным стандартным вызовом `pdm_score` со свежим metric cache. Baseline проверяется повторно до и после оценки остальных. Если ошибку нельзя устранить, запуск заканчивается с ошибкой без `report.json`.

Для прямого сравнения с CSV воспроизведённого v2-прогона при его наличии:

```bash
docker exec \
  -e DRIVEMEM_REFERENCE_CSV=/workspace/exp_v2/model_ckpt/test-drivesuprim_r34-epdms/2026.10.24.15.13.06/2026.10.24.16.41.17.csv \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_oracle_pilot.sh score
```

Эта команда заменяет предыдущую команду score: папка для одного и того же значения `DRIVEMEM_CANDIDATES` не перезаписывается. Скрипт сравнит доступные покадровые компоненты с CSV и остановится при расхождении. Полный EPDMS и two-frame comfort этой проверкой не воспроизводятся. Если файл недоступен, можно использовать команду без переменной и указать, что внешнее сравнение не выполнено.

## 3. Оценка всех 256 кандидатов

После успешного пилота и оценки времени можно использовать тот же экспорт:

```bash
docker exec -e DRIVEMEM_CANDIDATES=256 \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_oracle_pilot.sh score
```

Результат сохранится отдельно в `scoring_k256`. При желании в эту команду также передайте `DRIVEMEM_REFERENCE_CSV`. Повторная GPU-инференция не требуется, если checkpoint, код и конфигурация не менялись.

## Результаты

```bash
docker exec egorov_drivesuprim \
  cat /workspace/exp_v2/drivemem_oracle/pilot/scoring_k16/report.json

docker exec egorov_drivesuprim \
  head -n 10 /workspace/exp_v2/drivemem_oracle/pilot/scoring_k16/scene_summary.csv
```

`manifest.json` содержит конфигурацию, tokens, SHA256 checkpoint, словаря и пилотного скрипта. `candidate_metrics.csv` содержит оценку каждого кандидата; `scene_summary.csv` — baseline и локальные оракулы по сценам; `*_states.npz` — состояния после симуляции. `oracle_gain_mean_pp` и `safe_oracle_gain_mean_pp` выражены в процентных пунктах локальной метрики. Ограниченный оракул не допускает ухудшения NC/DAC/DDC/TLC/TTC относительно baseline по значениям scorer после human filter. Это не гарантия безопасности.

Если команда остановилась, сохраните полный traceback. В актуальной версии score можно продолжить той же командой: завершённые сцены сохраняются с hashes и пропускаются, а незавершённая считается заново. Код, checkpoint, конфигурация и reference CSV должны оставаться теми же. Результаты старой версии без `scoring_identity.json` не возобновляются; для новой версии используйте новый `DRIVEMEM_OUTPUT`. Разные `DRIVEMEM_CANDIDATES` создают независимые папки результатов, но использовать их как способ скрыть ошибку нельзя.

Для development-данных доступны `DRIVEMEM_SPLIT` и `DRIVEMEM_METRIC_CACHE`; требуется соответствующий подготовленный cache. `DRIVEMEM_SCENES` меняет количество сцен при экспорте и должен оставаться тем же для последующего score. Точную конфигурацию исходного полного v2-прогона необходимо дополнительно сверить с `manifest.json`.
