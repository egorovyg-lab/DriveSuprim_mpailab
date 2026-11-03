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
