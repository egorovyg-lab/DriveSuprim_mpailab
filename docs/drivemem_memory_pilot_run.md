# DriveMem: первый эксперимент с памятью на navtrain

Дата: 27 октября 2026 года. Это проверка минимального метода памяти, а не полный DriveMem с utility gate/калибровкой риска.

## Что реализовано

1. Потоковая инвентаризация ORIGINAL navtrain по одному log-файлу без загрузки изображений. Сегменты с одинаковым префиксом дата/время/машина объединяются в session_id. Удаляются дубли session/timestamp. Сессии официального navtest полностью исключаются.
2. Детерминированные роли memory / fit / development / calibration / test. Разделение целыми сессиями, без обращения к quality labels. Сцены разнесены на 10 секунд. Eligible sessions должны поддерживать максимальное запрошенное число сцен, поэтому это пилот на достаточно длинных сессиях, не равномерная выборка всех кадров датасета.
3. Собственный metric cache только для выбранных сцен. Старые oracle-результаты и cache не перезаписываются. Используется текущий MetricCacheProcessor репозитория с записанными hashes; нельзя молча объединять его результаты с результатами старого cache другого происхождения.
4. GPU-экспорт всех 256 финальных кандидатов, финальных heads и scores. Scene embedding — global average output frozen downscale_layer до positional embedding. Hook только читает выход; проверяется одинаковый baseline с выключенным/включённым capture. Сохраняется ego status текущего и предыдущего доступных входных кадров. Visual embedding относится к текущему кадру; это пока простой scene-memory baseline, не полноценная модель временных эпизодов.
5. Метки доноров и fit: baseline + top-8 по model score + 8 случайных кандидатов из остальных. Random выбор фиксирован по token/seed. Каждый кандидат оценивается стандартным pdm_score со свежим cache, отдельно без фильтра и с фильтром. Сохраняются компоненты, weighted vectors, multiplicative aggregate, raw progress и локальные баллы. Возможные расхождения filtered скаляров и агрегатов не исправляются скрыто.
6. Memory переносит residuals между исходом и прогнозами голов из пяти совместимых прошлых сцен. Доноры совместимы по driving command; расстояние использует frozen visual embedding и ego velocity. Геометрическое ядро сопоставляет действия; чужая траектория не копируется напрямую.
7. No-memory контроль — ridge regression для residuals метрик на признаках сцены, ego status, геометрии, исходных прогнозах и взаимодействиях context/action. Он получает все доступные memory+fit labels. Проекция, нормировка и веса обучаются только на этих данных. Ridge — первый параметрический контроль; перед итоговой статьёй потребуется проверить более сильные no-memory модели.
8. Оценка пяти методов: baseline, no_memory_ridge, random_memory, similarity_memory, similarity_threshold. Порог similarity берётся из 75%-квантили расстояний fit → memory, не из исходов development/test. Все выбирают из полного пула 256. На запросе scorer вызывается только для union выбранных действий после фиксации всех решений.

Вмешательство — реальная смена кандидата. Harm — ухудшение хотя бы одного raw NC/DAC/DDC/TLC/TTC относительно baseline. Отдельно считаются падение качества и абсолютные нарушения. Охват, harm и выигрыш публикуются вместе; это эмпирическая проверка, а не гарантия риска.

## Объём по умолчанию

| Роль | Сессии | Сцены/сессию | Сцен |
|---|---:|---:|---:|
| memory | 24 | 8 | 192 |
| fit | 16 | 4 | 64 |
| development | 8 | 4 | 32 |
| calibration | 8 | 4 | 32 |
| test | 8 | 4 | 32 |
| Всего | 64 | | 352 |

Проверить наличие не менее 64 подходящих navtrain-сессий. Если их не хватает, inventory завершится с сообщением и не будет автоматически подменять набор. Для M/F получается примерно 4096 candidate labels, каждый с raw/filtered оценкой; точная длительность зависит от подготовки cache и нагрузки. Потребуются данные `navsim_logs/trainval`, соответствующие `sensor_blobs/trainval`, карты и прежний R34 checkpoint. Если training dataset отсутствует, inventory сообщит путь; один navtest недостаточен.

Calibration и test заранее зарезервированы. `all` не оценивает на них методы и не создаёт их query labels. Они могут иметь подготовленный cache и frozen входные признаки, которые не используются для обучения. Независимость от pretraining опубликованного DriveSuprim пока не доказана; эти роли независимы от обучения нового модуля, но не являются основанием для conformal-гарантии.

## Передача через Git

Передать все новые файлы:

```text
navsim/agents/drivemem.py
navsim/planning/script/run_drivemem_memory.py
navsim/planning/script/config/pdm_scoring/default_drivemem_memory.yaml
scripts/drivesuprim/evaluation/run_drivemem_memory.sh
scripts/drivesuprim/evaluation/test_drivemem_memory.py
docs/drivemem_memory_pilot_run.md
```

Также желательно сохранить `docs/drivemem_revised_research_plan.md`. Из первого пилота в репозитории должны оставаться экспортные поля модели и `InferConfig.export_candidates`. Для нового пакета дополнительная правка базовых весов/архитектуры не нужна.

## 1. Проверить тесты и конфигурацию

После git pull на сервере:

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh check
```

Проверяются CPU-алгоритмы, утечки ролей, запросы без доступа к labels, сохранение no-op baseline, выбор из всех 256, детерминированность, fit/evaluate/resume на синтетическом scorer. На сервере дополнительно выполняются тест композиции Hydra и `--cfg job --resolve` до тяжёлого запуска.

## 2. Инвентаризация и разбиение

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh inventory
```

Выведет число сессий/сцен и распределение ролей. Сохранит inventory.csv, inventory_report.json, splits.json, split_identity.json и identity.json. Ни GPU-инференция, ни candidate scoring на этой стадии не выполняются.

По умолчанию:

```text
/workspace/exp_v2/drivemem_memory/pilot_v1
```

## 3. Полный development-пилот

После успешных check/inventory:

```bash
docker exec -d -w /workspace/DriveSuprim egorov_drivesuprim bash -c \
  'bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh all > /workspace/exp_v2/drivemem_memory_pilot.log 2>&1'
```

Последовательность: существующее разбиение → cache → GPU export → labels M/F → fit frozen ridge → evaluate development. Завершение docker exec -d означает запуск, а не успешное окончание расчёта.

```bash
docker exec egorov_drivesuprim \
  tail -n 50 /workspace/exp_v2/drivemem_memory_pilot.log
```

Каждый шаг сохраняет результаты по сценам. Повтор `all` продолжит этапы, не переписывая frozen fit; если GPU-экспорт ещё не выполнен, GPU всё ещё потребуется. Запускать один процесс на root, не менять код/конфигурацию/checkpoint между стадиями. Hash-проверка остановит смешивание версий.

Для отдельных стадий вместо all доступны cache, export, label, fit, evaluate. Оценка development может продолжаться командой evaluate без повторного GPU-экспорта.

## 4. Отчёт

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh report
```

Основные файлы:

- `evaluation_development/report.json`: пять методов, raw/filtered качество, paired gain CI по сессиям, coverage, число interventions/harm, conditional harm и quality degradation, абсолютные нарушения;
- `evaluation_development/per_scene.csv`: решения, похожесть/доноры, компоненты raw/filtered, Δ;
- `evaluation_development/<token>.json`: frozen decisions и результаты union действий;
- `labels/<token>.json`: simulator-supervised memory/fit targets и происхождение features;
- `ridge.npz`, `fit.json`: frozen параметрический контроль, порог и hashes;
- `features/<token>.npz`: только доступные входные признаки/предсказания, без будущих outcomes.

Сначала сравнить memory с baseline и no_memory_ridge. Если memory ухудшает результат, это допустимый результат пилота; не подбирать правило по test, не называть любые отказы безопасными и не обещать положительный результат заранее. Главный вопрос: есть ли полезные и вредные memory-вмешательства и зависит ли их знак от similarity.

## Test, calibration и варианты банка

Test запускается отдельной командой `run_drivemem_memory.sh test` после фиксации метода и анализа development. После первого просмотра test результаты становятся частью истории использования этого набора. Calibration пока не используется: utility/risk gate не реализован.

По умолчанию bank **enriched**: прошлые альтернативные действия оценены в симуляторе. Это дополнительное supervision, а не естественный поток реальных исполненных действий. Для банка только выбранных baseline-действий создать новый root и передавать во ВСЕ команды:

```bash
docker exec \
  -e DRIVEMEM_MEMORY_MODE=executed_only \
  -e DRIVEMEM_MEMORY_ROOT=/workspace/exp_v2/drivemem_memory/executed_only_v1 \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_memory.sh inventory
```

Затем те же env при all/report. Здесь исходы выбранных действий тоже получены симуляцией. No-memory baseline использует ту же доступную донорскую supervision, что и memory-методы.

Количество сессий меняют DRIVEMEM_MEMORY_SESSIONS, DRIVEMEM_FIT_SESSIONS, DRIVEMEM_DEV_SESSIONS, DRIVEMEM_CAL_SESSIONS, DRIVEMEM_TEST_SESSIONS; число сцен — DRIVEMEM_DONOR_SCENES, DRIVEMEM_FIT_SCENES, DRIVEMEM_QUERY_SCENES. Любая смена настроек требует нового root. Фиксированный seed не гарантирует тот же набор при изменившемся содержимом датасета; сохраняется фактический manifest.

Ограничения этой версии: frozen bank, exact retrieval без FAISS, простая коррекция residuals, ridge-контроль, локальная метрика без двухкадрового комфорта. Это код для первой научной развилки, не финальная реализация статьи. Хронологическое обновление памяти, обучаемая оценка полезности, риск-калибровка, полный EPDMS и второй planner идут следующими этапами после результата пилота.
