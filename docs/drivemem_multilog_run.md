# DriveMem: проверка локального оракула на разных логах

Дата: 27 октября 2026 года.

## Дизайн

Выбрать 32 log_name из полного navtest, по две сцены на лог, с разницей времени не менее 10 секунд. Используется seed 20261027. Лог `2021.06.03.12.02.06_veh-35_01100_01227`, изученный в первом пилоте, исключён. Логи выбираются равновероятно из поддерживающих заданное число разнесённых сцен; затем выбираются допустимые времена. Исходы кандидатов при отборе не используются. Это выборка с равным весом логов, а не случайная выборка всех кадров navtest. Логи могут быть фрагментами связанных поездок: 10 секунд не гарантируют независимость событий, поэтому статистика группируется по log_name.

Проверяются все финальные 256 кандидатов на 64 сценах. Оценка каждого кандидата использует отдельный стандартный вызов `pdm_score` со свежим cache. Сохраняются проверки export on/off, повтор baseline до и после остальных кандидатов и сравнение компонентов с CSV воспроизведения. Для оценки времени ориентир прошлого запуска: 868 секунд на 8 сцен; 64 сцены могут потребовать примерно два часа последовательного CPU-расчёта плюс время экспорта. Это приблизительная оценка.

Результат описывает верхнюю границу локальной метрики без two-frame comfort. Он не является измерением пользы памяти, её отрицательного переноса, полного EPDMS либо closed-loop безопасности. NAVSIM navtest здесь используется для описательного исследования исходной системы; обучать модуль памяти или подбирать его пороги по этим меткам нельзя.

## Передача через Git

В commit должны войти:

```text
navsim/evaluate/drivemem_oracle_utils.py
navsim/planning/script/run_drivemem_oracle_pilot.py
scripts/drivesuprim/evaluation/run_drivemem_oracle_pilot.sh
scripts/drivesuprim/evaluation/run_drivemem_multilog.sh
scripts/drivesuprim/evaluation/test_drivemem_oracle_utils.py
docs/drivemem_multilog_run.md
docs/drivemem_oracle_pilot_run.md
```

После commit/push и pull на сервере выполнить команды ниже. Модель и InferConfig уже должны содержать изменение `export_candidates` из первого пилота. Дополнительная правка модели для этого этапа не нужна.

## Проверка утилит в контейнере

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  python -m unittest discover -s scripts/drivesuprim/evaluation \
  -p test_drivemem_oracle_utils.py -v
```

Шесть CPU-тестов проверяют детерминированность отбора, исключение пилотного лога, временные интервалы, отсутствие повторов, constrained oracle, кластерный интервал, возобновление после искусственной ошибки и обнаружение повреждённых результатов. Они не заменяют реальную проверку NAVSIM на сервере.

## Полный запуск в терминале

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_multilog.sh all
```

Для фонового запуска вместо этой команды:

```bash
docker exec -d -w /workspace/DriveSuprim egorov_drivesuprim bash -c \
  'bash scripts/drivesuprim/evaluation/run_drivemem_multilog.sh all > /workspace/exp_v2/drivemem_multilog.log 2>&1'

docker exec egorov_drivesuprim tail -n 40 /workspace/exp_v2/drivemem_multilog.log
```

Код сначала выполняет export, затем score. Завершение `docker exec -d` означает запуск фонового процесса, а не успешное окончание эксперимента. Отчёт со статусом complete появляется после обработки всех сцен.

Новая папка по умолчанию:

```text
/workspace/exp_v2/drivemem_oracle/multilog32x2_seed20261027
```

Список выбранных сцен и параметры отбора записаны в `manifest.json` до оценки. Если доступных логов недостаточно либо для выбранной сцены нет metric cache, запуск завершается ошибкой, не заменяя её более удобной сценой.

## Возобновление после остановки

Если export завершился, а CPU-расчёт остановился, выполнить:

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_multilog.sh score
```

Завершённые сцены пропускаются после проверки hashes. Незавершённая сцена считается заново. Данные состояния пишутся атомарно, после чего создаётся marker `*_done.json`. Конфигурация, исходники, checkpoint и reference CSV должны оставаться теми же; не делать pull между export и score/resume. Одновременно запускать два score в одной папке нельзя.

После неудачного export потребуется новый `DRIVEMEM_OUTPUT`; возобновление GPU-экспорта в этом варианте не реализовано. Старый пилот из восьми сцен сохраняется в отдельной папке; новая версия скрипта намеренно не смешивает с ним новые scores из-за проверки hashes.

## Отчёт

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_multilog.sh report
```

Выводит `report.json` и `log_summary.csv`. В папке `scoring_k256` также находятся:

- `scene_summary.csv`: baseline/oracle, выигрыш, rank лучшего кандидата, прогнозы NC/TTC и их оценки scorer, число лучших альтернатив, резерв top-8/16/32/64/256;
- `candidate_metrics.csv`: все компоненты каждого кандидата и его финальные прогнозы NC/TTC;
- `*_states.npz`: состояния симуляции, `*_done.json`: результаты сцены для resume;
- `scoring_identity.json`: идентичность export и reference CSV.

В report важны `logs`, `scenes`, `logs_with_improvement`, `oracle_gain_mean_pp`, `safe_oracle_gain_mean_pp`, `oracle_gain_pp_cluster_ci95`, `safe_oracle_gain_pp_cluster_ci95`, `topk_mean_gain_pp`. Интервал рассчитывается paired cluster bootstrap по log_name, 2000 повторов. `scoring_seconds` означает время текущего вызова score, включая чтение сохранённых результатов при resume, а не сумму времени всех попыток.

Сначала проверить, что выгода встречается в разных логах и не определяется одной поездкой. Затем сравнить ограниченный и неограниченный оракулы и резерв при разных top-K. Эти результаты нужны для выбора следующего исследовательского этапа; один высокий средний балл не доказывает полезность памяти.

По умолчанию reference CSV взят из успешного v2-прогона:

```text
/workspace/exp_v2/model_ckpt/test-drivesuprim_r34-epdms/2026.10.24.15.13.06/2026.10.24.16.41.17.csv
```

Чтобы задать другой путь, передать `docker exec -e DRIVEMEM_REFERENCE_CSV=...` во все стадии. Пустое значение отключает сравнение, что будет явно указано в report. Отсутствующий заданный файл вызывает ошибку.

Для другого объёма доступны `DRIVEMEM_LOGS`, `DRIVEMEM_SCENES_PER_LOG`, `DRIVEMEM_MIN_GAP_S`, `DRIVEMEM_SEED`. При их изменении задавать новую папку через `DRIVEMEM_OUTPUT` и передавать те же значения в export/score/report.
