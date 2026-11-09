# Единый train bundle: старые 112 + новые 287 сцен

Сбор новой части завершён. До следующего обучения объединяем совместимые
признаки и сохранённые raw/filtered labels. Работа выполняется на CPU, без
GPU, scorer calls, новых изображений или открытия fresh holdout.

## Запуск внутри контейнера

После commit/push/pull:

```bash
cd /workspace/DriveSuprim
bash scripts/drivesuprim/evaluation/run_selector_bundle.sh check
bash scripts/drivesuprim/evaluation/run_selector_bundle.sh build
```

`build` проверяет старые manifests и accepted functional sidecar, новый
train-only collector, его label identity и checksums, совместимость checkpoint,
vocabulary, teacher/scorer config и declared numerical profile. Затем копирует
только разрешённые числовые признаки и top16 labels в отдельный bundle.
Дубликаты token, несовместимые shapes и пересечения с protected/holdout sessions
вызывают остановку. Existing collection/experiments остаются read-only.

Повторный аудит без пересборки:

```bash
bash scripts/drivesuprim/evaluation/run_selector_bundle.sh audit
```

## Выход

Root: `/workspace/exp_v2/drivemem_memory/selector_training_bundle_v2`.
Ожидаемые counts: 399 scenes, 22 sessions. Числа проверяются по фактическим
metadata источников, не подставляются вместо измеренных данных.

- `identity.json`: исходные hashes, код, protected sessions и counts.
- `examples/*.npz`: frozen inference inputs и отдельные label arrays.
- `examples_index.json`: scene/session/origin metadata и hashes примеров.
- `vocab.npz`, `bundle_manifest.json`: общий vocabulary и целостность bundle.
- `audit.json`, `audit_per_scene.json`: полезные альтернативы, oracle reserve,
  DAC/NC/DDC/TLC/TTC support, распределение по поездкам и городам.

Считаются микро-средние по сценам и макро-средние по целым сессиям. Число
кандидатных строк не считается числом независимых примеров. Полезная замена:
raw PDM gain >=0.005 без ухудшения любого из пяти safety-компонентов относительно
baseline. Это относительное условие, не абсолютная безопасность. Stored raw PDM
не восстанавливается из filtered components.

Reader возвращает `sample` (явный allowlist input keys), `label_raw`,
`label_filtered`, `label_indices` и `metadata` раздельно. Список INPUT_KEYS
не содержит outcomes/scene IDs/session IDs. V2 training должен использовать
этот reader и не брать все NPZ arrays автоматически как признаки.

Для каждого примера сохранён origin `legacy`/`new` и feature protocol. Новый
сбор не переименовывается в историческую functional acceptance: происхождение
двух частей явно различается. Общий vocabulary, config и numerical profile
проверяются, но это не заявление о равенстве всех исторических float logits.

Следующее обучение: matched MLP v1 на расширенных данных как контроль эффекта
датасета и MLP с одной согласованной utility-оценкой для ranking/gain. Выбор
параметров — только по train sessions. Старые 40 development scenes остаются
диагностикой; новые четыре holdout sessions сохраняются до фиксации модели.
Этот bundle stage модели не обучает.
