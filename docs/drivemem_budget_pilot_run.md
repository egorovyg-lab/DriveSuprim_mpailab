# DriveMem: пилот с ограничением объёма архивов

Исходный random-session набор 352 сцен требует до 449 GB компактных архивов либо около 790 GB полных camera shards. Уменьшение числа случайных сцен само по себе не гарантирует существенного сокращения загрузки.

Новый preparer сначала выбирает небольшое множество camera shards по стоимости и покрытию navtrain-сессий. Затем случайно распределяет доступные сессии по пяти ролям и выбирает кадры с интервалом не менее 10 секунд. Каждая сессия целиком принадлежит одной роли; официальные navtest-сессии исключены исходным проверенным inventory. При выборе не читаются candidate metrics, labels и результаты оценки.

По умолчанию лимит — **20 GB** (десятичные GB) суммарных сжатых архивов. Это объём уникальных архивов, без HTTP overhead и повторной передачи при сбоях. Значительно меньше исходных 449 GB; точный размер определяется командой plan, до загрузки изображений. Поиск — детерминированная эвристика покрытия с несколькими начальными вариантами; оптимальность стоимости не гарантируется. Если подходящий набор не найден, лимит и количество сессий автоматически не увеличиваются/уменьшаются.

Набор по умолчанию: memory 6×8=48 сцен, fit 4×8=32, development 3×8=24, calibration 2×8=16, test 2×8=16. Всего **17 сессий / 136 сцен**. Из каждой сессии выбирается один подходящий сегмент. Используются исходные восемь камер на четырёх past/current кадрах; будущие изображения и LiDAR не нужны.

Этот набор предназначен для технической проверки retrieval, corrections, no-memory baseline и измерения вмешательств. Отбор архивов по стоимости меняет распределение данных; три development-сессии дают слабую статистическую точность. Результаты нельзя переносить на полный navtest или напрямую сравнивать с +5,73 pp oracle на другом наборе. Calibration/test пока резервируются; независимость от pretraining не установлена.

## Обновить код

Нужны новые файлы:

- `download/prepare_drivemem_budget.py`;
- `download/fetch_drivemem_budget_images.py`;
- `scripts/drivesuprim/evaluation/run_drivemem_budget.sh`;
- `scripts/drivesuprim/evaluation/test_drivemem_budget.py`;
- этот runbook.

Также изменён `download/plan_drivemem_images.py`: manifest теперь содержит точное соответствие log → images. Основной экспериментальный код и его зафиксированные hashes не менялись.

После commit/push локально и pull на сервере:

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh check

docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh plan
```

Wrapper сам задаёт writable `OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data`, если переменная не установлена. Если в контейнере она уже указывает на `/workspace/dataset`, явно передавать `-e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data` во все команды либо сохранять исходные data paths через `DRIVEMEM_LOGS_PATH`/`DRIVEMEM_SENSOR_PATH`. Для приведённого существующего пилота рекомендуется явно передавать `-e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data` во все команды.

`plan` читает проверенный inventory из `/workspace/exp_v2/drivemem_memory/pilot_v1`, получает только небольшие официальные manifests и создаёт отдельный `/workspace/exp_v2/drivemem_memory/budget20gb_v1`. Старые 352 сцены, splits и cache сохраняются. Отдельные камеры ещё не загружаются. Новый identity наследует конфигурацию/код/checkpoint исходного пилота с явным изменением root и размеров пяти ролей; затем исходный main проверяет его через inventory. Несовпадение конфигурации останавливает запуск до больших загрузок.

В отчёте plan проверить `compressed_GB <= limit_GB`, `selected_sessions: 17`, `selected_scenes: 136` и пять ролей. Список архивов и точные images уже зафиксированы. Один root обслуживается одним процессом.

## Загрузка изображений

После успешного plan:

```bash
docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh images

docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh images-status
```

Архивы скачиваются по одному с продолжением `.part`. Перед извлечением проверяются полный размер и SHA256 из official Git-LFS metadata. Извлекаются только точные requested images, без `extractall`, symlinks и перезаписи отличающихся существующих файлов. После проверки полного покрытия каждого архива сохраняется checksummed receipt. По умолчанию сжатые архивы остаются в общем кэше `$OPENSCENE_DATA_ROOT/.downloads/drivemem_camera`, чтобы расширять число кадров внутри этих архивов без повторной передачи. Нужны место под все выбранные сжатые архивы (до 20 GB), извлечённые картинки и резерв. Можно задать `DRIVEMEM_IMAGE_ARCHIVE_DIR`; при нехватке диска `DRIVEMEM_DISCARD_ARCHIVES=1` удаляет каждый архив после установки, но будущий новый набор может потребовать повторной загрузки. Отдельный отчёт проверяет SHA256 каждой установленной картинки. Остановка из-за сети не требует нового root: повторить images. Автоматические curl retries выключены, чтобы не скрывать дополнительный трафик.

Успех: `status: complete`, `verified_images == required_images`, `missing_images: 0`. Не запускать обычный image-plan поверх бюджетного archive_plan: его изменение будет обнаружено, и fetch остановится. Если после сбоя plan создал неполный root, сохранить его для диагностики и выбрать новый root; завершённый plan повторяется без изменения splits.

## Запуск memory

```bash
docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh cache

docker exec -d -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim bash -c \
  'bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh all > /workspace/exp_v2/drivemem_budget_pilot.log 2>&1'

docker exec egorov_drivesuprim tail -n 50 /workspace/exp_v2/drivemem_budget_pilot.log
```

Новый cache создаётся для новых 136 сцен. `all` продолжает cache, затем export → labels M/F → fit → development. Перед all/export wrapper проверяет полное покрытие изображениями. Повтор all продолжает уже завершённые стадии. Calibration/test команда all не оценивает.

```bash
docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh report
```

## Ещё меньший бюджет

Лимит и размеры задаются env при plan и сохраняются неизменными во всех следующих командах. Например, техническая smoke-проверка до **5 GB**, по одной сессии на роль и четыре сцены на сессию (20 сцен):

```bash
docker exec \
  -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -e DRIVEMEM_MEMORY_ROOT=/workspace/exp_v2/drivemem_memory/budget5gb_smoke_v1 \
  -e DRIVEMEM_MAX_DOWNLOAD_GB=5 \
  -e DRIVEMEM_MEMORY_SESSIONS=1 -e DRIVEMEM_FIT_SESSIONS=1 \
  -e DRIVEMEM_DEV_SESSIONS=1 -e DRIVEMEM_CAL_SESSIONS=1 -e DRIVEMEM_TEST_SESSIONS=1 \
  -e DRIVEMEM_BUDGET_SCENES=4 \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh plan
```

С теми же env заменить plan на images, images-status, cache, all или report. Этот профиль проверяет исполнение кода; одна сессия на роль непригодна для содержательной статистики эффекта или вреда. Наличие подходящего набора даже для 5 GB предварительно не обещается — это проверяет metadata-only plan. Дополнительная загрузка metadata не требуется.
