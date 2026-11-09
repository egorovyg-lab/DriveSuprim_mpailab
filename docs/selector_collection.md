# Сбор признаков и top16 labels расширенного train

Вход — завершённая подготовка `selector_data_expansion_v2`: 287 новых train
scenes и отдельно 64 holdout scenes. Сборщик обрабатывает только train. Старые
112 M/F scenes и accepted functional features сохраняются. После сбора ожидается
399 размеченных обучающих сцен; объединение с legacy для обучения v2 — следующий
этап. Старый `trajectory_selector_v1` loader с квотой 112 не запускать на этом root.

## Запуск внутри контейнера

После commit/push/pull новых файлов, внутри `/workspace/DriveSuprim`:

```bash
cd /workspace/DriveSuprim
bash scripts/drivesuprim/evaluation/run_selector_collection.sh check
bash scripts/drivesuprim/evaluation/run_selector_collection.sh init
bash scripts/drivesuprim/evaluation/run_selector_collection.sh cache
bash scripts/drivesuprim/evaluation/run_selector_collection.sh export-pilot
```

`cache` работает на CPU с установленными maps. `export-pilot` требует CUDA,
обрабатывает четыре сцены с покрытием имеющихся городов. После успешного пилота:

```bash
bash scripts/drivesuprim/evaluation/run_selector_collection.sh export
bash scripts/drivesuprim/evaluation/run_selector_collection.sh status
bash scripts/drivesuprim/evaluation/run_selector_collection.sh label-pilot
```

До labels должны быть завершены cache и export всех 287 train scenes. Pilot
размечает до восьми ещё не завершённых сцен, сохраняет время в
`last_label_timing.json` и поддержку safety events в `support_report.json`.
Повторный pilot продолжит со следующих незавершённых сцен.

Полная CPU-разметка:

```bash
nohup bash scripts/drivesuprim/evaluation/run_selector_collection.sh label \
  > /workspace/exp_v2/selector_collection_label_v2.log 2>&1 &
tail -n 40 /workspace/exp_v2/selector_collection_label_v2.log
```

Отчёт:

```bash
bash scripts/drivesuprim/evaluation/run_selector_collection.sh report
```

Один процесс на collection root. Завершённые cache/features/labels не
пересчитываются; их hashes проверяются. Частичная label-сцена возобновляется
после повторной проверки baseline. `SELECTOR_COLLECTION_GPU` выбирает один GPU,
`SELECTOR_COLLECTION_THREADS` — CPU threads. Новых скачиваний нет.

## Что фиксируется

Выход: `/workspace/exp_v2/drivemem_memory/selector_collection_v2`.
`init` проверяет acquisition/изображения/checkpoint/source hashes; сохраняет
train-only `splits.json`, `train_images.json`, `source_config.json`, `identity.json`.
Holdout отсутствует в cache/export/label membership.

Для каждой сцены первый native forward фиксируется в `native/` до исходов.
Его ordered 256 IDs затем используются в двух текущих replay forwards:
без capture и с capture. Они должны совпасть бит-в-бит; относительно native
требуются прежние functional budgets, тот же baseline/trajectory и top16 set.
IMI drift относительно native записывается как диагностический, критерии заданы
до сбора. Это новый prospective dataset protocol, не повторный подбор допуска
под уже размеченные сцены. Native pool drift последующих forwards записывается.

Из того же capture forward сохраняются scene embedding, ego, 256 decoder states,
все metric/IMI heads и scores. Привязка hidden states дополнительно проверяется
повторным применением выходных голов. При parity/alignment failure сохраняется
`failure.json`; дальнейший export останавливается для разбора, без смены
кандидатов или допусков. Отсутствие CUDA не повреждает завершённые файлы.

Перед labels фиксируются все feature/cache/vocab hashes. Каждый кандидат
оценивается отдельно штатным local PDM scorer с raw/filtered вариантами.
Храним сам `raw pdm_score`, не восстанавливаем его из filtered components.
Report содержит baseline/oracle reserve на train и число DAC/NC/DDC/TLC/TTC
ухудшений/улучшений, отдельно количество затронутых сцен и сессий.
Это аудит train, не оценка новой модели или калиброванной безопасности.

При изменении кода, checkpoint, входов или numerical runtime прежний root
сохраняется, для нового протокола нужен другой `SELECTOR_COLLECTION_ROOT`.
GPU-путь локально без серверных данных не проверяется; обязательный следующий
шаг — export-pilot в целевом контейнере. Unit tests используют искусственные
features/scorers и автоматически удаляемые временные каталоги.
