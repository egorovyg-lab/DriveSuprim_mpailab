# Новые сессии в локальных архивах: frozen-model validation

Проверяется зафиксированный B3 из `experiments/action_transfer_v2`: `h=0.0010552772409836734`, beta=.5, пять доноров, исходный M-bank и frozen full-embedding ridge. Другие заранее определённые B0/B1/B2, scene-only и три random seeds остаются контролями. Обучения и нового выбора параметров нет.

Набор: четыре новые доступные сессии из исходных локальных archives, по четыре сцены с gap≥10s, всего 16 сцен. Отбор использует только metadata. Все сессии текущего 17-session пилота, исходные резервные C/T и official navtest исключены. Это предварительная проверка на новых сессиях; archive-selection bias остаётся, число сессий мало, независимость от pretraining DriveSuprim не установлена.

Новый root: `/workspace/exp_v2/drivemem_memory/validation_cached4_v1`. Исходный data root и camera cache общие. Сессии помечены `development` ради совместимости exporter; выделенный runner использует их только для применения frozen models. В этом наборе нет своих M/F/C/T; исходный M-bank загружается read-only из `budget20gb_v1`. Обычный memory `all/fit` к этому root неприменим — использовать только специальный launcher.

После commit/push и pull:

```bash
docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_cached_validation.sh check

docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_cached_validation.sh plan
```

`plan` получает только маленький официальный mapping JSON, проверяет frozen identities и создаёт фиксированное разбиение 4×4, hashes исходных logs и точный image manifest. Готовые исходные артефакты не изменяются. Новая identity наследует тот же agent/scorer/checkpoint с новым root и нулевыми локальными M/F/C/T; затем обычный main проверяет Hydra-конфигурацию. Если доступно не ровно четыре новых сессии, preparer останавливается для проверки набора. Повтор успешного plan использует frozen manifest. Неполный root после сбоя подготовительного этапа сохранять для диагностики и выбирать новый `DRIVEMEM_VALIDATION_ROOT`.

```bash
docker exec -d -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim bash -c \
  'bash scripts/drivesuprim/evaluation/run_drivemem_cached_validation.sh all >> /workspace/exp_v2/drivemem_cached_validation.log 2>&1'

docker exec egorov_drivesuprim tail -n 60 /workspace/exp_v2/drivemem_cached_validation.log
```

Стадии: offline extraction только нужных изображений из local tgz → новый cache → GPU export → frozen-model scoring union выбранных действий. Перед распаковкой проверяется полный SHA256 каждого нужного архива; сканирование архивов требует времени и дисковых операций. При отсутствии полного архива `--offline` запрещает запуск curl: никакой image transfer автоматически не начинается. Архивы сохраняются для повторного использования. Метаданные manifest — единственный маленький сетевой запрос этапа plan.

**GPU требуется для нового export 16 сцен.** Cache и scorer работают на CPU. Если контейнер снова теряет GPU, экспорт остановится; готовые файлы можно продолжить после восстановления CUDA. Новая разметка только union baseline/выбранных методов, без полного oracle. Сначала все decisions фиксируются без outcomes, затем каждый action отдельно оценивается original evaluator с fresh cache/reference pair; baseline повторяется для parity. После каждого действия есть resumable JSON. C/T outcomes не открываются.

```bash
docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_cached_validation.sh report
```

Другие стадии: images, images-status, cache, cache-status, export, evaluate. Повтор all использует готовые extraction receipts/cache/features/действия, модели не переобучает; baseline repeats остаются. Один процесс на root.

Модель задаётся `DRIVEMEM_FROZEN_MODEL_ROOT`, исходный bank — `DRIVEMEM_FROZEN_SOURCE_ROOT`, новый набор — `DRIVEMEM_VALIDATION_ROOT`, camera cache — `DRIVEMEM_IMAGE_ARCHIVE_DIR`. Для текущего протокола seed и режим enriched зафиксированы launcher. Все пути/версия fit фиксируются plan, поэтому смена этих значений требует нового validation root. Новые runner/model/input hashes проверяются перед повтором scoring; identity вручную не редактировать.

Отчёты в `validation_cached4_v1/frozen_evaluation`: identity.json, decisions.json, scenes/<token>.json, per_scene.csv, by_session.csv и report.json. Это Local PDM, не full EPDMS; H относительный по пяти safety-компонентам, абсолютные нарушения сообщаются отдельно. После просмотра новых результатов эти сессии тоже считаются использованными: для следующего metadata-планирования передавать `--exclude-run /workspace/exp_v2/drivemem_memory/validation_cached4_v1`.

## Диагностика после отрицательной frozen-проверки

Сначала сохранить отрицательный результат и frozen параметры. Не менять h/beta по этой выборке и не объявлять B1 новым основным методом после просмотра validation. Для деталей по четырём вмешательствам:

```bash
docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_cached_validation.sh diagnose
```

Команда read-only выводит gain по сессиям, все component deltas primary и полный raw-вектор худшего случая. H=0 исключает ухудшения пяти заданных safety-компонентов, но не исключает потери progress/comfort/PDM.

Чтобы проверить резерв того же candidate pool, доступны oracle-plan/oracle. Используется отдельный `run_drivemem_validation_oracle.py`: original oracle runner, входящий в source hashes обученного v2, не меняется. Адаптер читает уже завершённые frozen-evaluation states и повторно использует их исходы. Эти новые labels — post-evaluation diagnostic; ни одна модель ими не обучается.

```bash
docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_cached_validation.sh oracle-plan

docker exec -d -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim bash -c \
  'bash scripts/drivesuprim/evaluation/run_drivemem_cached_validation.sh oracle >> /workspace/exp_v2/drivemem_cached4_oracle.log 2>&1'

docker exec egorov_drivesuprim tail -n 50 /workspace/exp_v2/drivemem_cached4_oracle.log

docker exec egorov_drivesuprim cat \
  /workspace/exp_v2/drivemem_memory/validation_cached4_v1/development_oracle/report_top16.json
```

По умолчанию k=16. Максимум 16×15 новых действий +16 baseline repeats, то есть 512 raw/filtered scorer calls; уже известные действия снижают стоимость. Выход также содержит top-8. GPU и новые изображения не нужны. Можно передать `DRIVEMEM_ORACLE_TOPK=8` для меньшего диагностического префикса, но по нему нельзя заключать об отсутствии резерва во всём top-16. Повтор или расширение k использует готовые outcomes. Пул 256 автоматически не считается. Отчёт называется development_oracle ради совместимости helper, но относится к новой 4-session cohort, не к исходным 24 V-сценам.

Развилка: заметный резерв top-16 при проигрыше B3 требует анализа переносимости residuals/scene representation и action support; малый резерв ограничивает интерпретацию отсутствия gain, но не отменяет измеренные ухудшения B3. Ни один исход не обосновывает самостоятельный положительный эффект gate: на этих 16 сценах среди нынешних B3 proposals нет улучшений, и oracle отказа может лишь вернуть baseline. Перед следующей итерацией определить новые данные для проверки и явно учесть, что эти четыре сессии теперь просмотрены.

## Результат oracle: проверить покрытие действий в памяти

Получено на cached4: baseline 85.6927, B3 −0.9579 п.п.; oracle top-8 +0.8451, top-16 +7.2464 п.п., safe oracle даёт тот же полный резерв. Top-8 содержит лишь около 11.7% резерва top-16, тогда как на исходном V было около 98%. Основная добавка текущего oracle появляется при разрешении model ranks 9–16.

Это не доказательство причины неудачи B3. Memory M размечена по схеме top-8 + random-8, поэтому дополнительная проверка должна различать локальные ranks, геометрию действий и контекст донорских исходов. Candidate rank зависит от сцены: query rank 12 может геометрически совпадать с donor rank 2. Отсутствие M labels с локальным rank 9–16 само по себе не доказывает отсутствия поддержки query-действий.

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  python -u scripts/drivesuprim/evaluation/audit_drivemem_action_support.py \
  /workspace/exp_v2/drivemem_memory/experiments/action_transfer_v2 \
  /workspace/exp_v2/drivemem_memory/validation_cached4_v1
```

Read-only, без GPU/загрузок/simulator. Выводит M label coverage по полосам 1–8 / 9–16 / 17–256, числа измеренных компонентов ниже 1, недостаток M top-16 labels и цену возможного дополнения (максимум 384 новых M actions +48 baseline repeats =864 raw/filtered calls; скрипт их не выполняет). Для обоих просмотренных query-наборов показывает резерв по сессиям и поддержку baseline/oracle действий: расстояния в единицах фиксированного h, ESS и точные совпадения геометрии среди retrieved-5 и всех command-compatible M эпизодов.

Следующее решение после аудита: если полезные действия плохо покрыты во всём M, проверить равномерное завершение M top-16 в отдельном sidecar с фиксированным h, равной дополнительной supervision для ridge и новыми данными для последующей проверки. Если покрыты во всём M, но не в retrieved-5, исследовать выбор доноров. Если геометрически покрыты и в retrieved-5, а их residuals не переносятся, упор на M coverage недостаточен — нужен анализ контекста/targets/представления. Геометрическое совпадение не гарантирует одинаковых исходов. C/T и query oracle labels не используются как memory supervision. Нельзя использовать увиденный oracle rank как признак модели во время решения.
