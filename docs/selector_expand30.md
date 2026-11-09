# Второе расширение: до 30 ГБ дополнительно

Пользователь разрешил ещё 30 ГБ сжатых image archives сверх уже использованных
30 ГБ. Этот этап добавляет данные к готовому bundle 399 сцен / 22 поездки.
Архивы выбираются совместно по metadata и стоимости: приоритет новым поездкам
с минимум 16 сценами. Повторы между архивами не считаются новыми поездками.
Используется детерминированная multistart greedy эвристика, без доказательства
глобальной оптимальности. Объём и число поездок показываются после планирования,
не обещаются заранее. Архивная выборка не является репрезентативным benchmark.

## Запуск внутри контейнера

После commit/push/pull:

```bash
cd /workspace/DriveSuprim
bash scripts/drivesuprim/evaluation/run_selector_expand30.sh check
bash scripts/drivesuprim/evaluation/run_selector_expand30.sh plan
```

`plan` использует сохранённые официальные manifests, не скачивает изображения
или JSON и не открывает feature/label arrays. Он печатает новые сцены, новые
поездки, добавление в старые поездки, список выбранных архивов, дополнительный
объём и будущую стоимость разметки в scorer calls. Все 399 предыдущих сцен
являются anchors. Новые сцены отстоят от них и друг от друга минимум на 10 секунд;
суммарный лимит — 128 сцен на поездку. Новая поездка должна дать минимум 16 сцен.
Четыре fresh holdout и остальные защищённые поездки исключены целиком. Новые
сцены выбираются по первому допустимому времени, без использования outcomes.
Дополнительные holdout в этом раунде не назначаются.

Полная подготовка данных, включая download/extraction, cache, GPU export,
top16 labels и новый объединённый bundle:

```bash
nohup bash scripts/drivesuprim/evaluation/run_selector_expand30.sh all \
  > /workspace/exp_v2/selector_expand30_v1.log 2>&1 &
tail -n 60 /workspace/exp_v2/selector_expand30_v1.log
```

`all` готовит данные и не запускает обучение селектора. План уже содержит
разрешённый лимит <=30 ГБ: повторного подтверждения объёма в этом пределе не нужно.
Лимит относится к уникальным новым сжатым архивам, без HTTP overhead и повторного
трафика при сетевых сбоях. Скачанные архивы сохраняются; существующие используются
повторно. Перед извлечением проверяется SHA256. Исчезнувший/испорченный cached
архив вызывает остановку перед новыми загрузками. Учтены только полностью
сохранённые архивы; изображения без архивов не участвуют в поиске покрытия,
но при подготовке выбранных сцен существующие совпавшие изображения сохраняются.

GPU нужен для экспорта признаков; cache, labels и сборка bundle работают на CPU.
На GTX 1080 используется прежний экспорт с проверками повторяемости и привязки
кандидатов. Ошибки CUDA/parity сохраняют готовые файлы. Не удалять failure.json
для обхода проверки. Сбор и длительное CPU обучение конкурируют за ресурсы CPU;
можно выполнять их последовательно.

Для отдельных стадий доступны:

```bash
bash scripts/drivesuprim/evaluation/run_selector_expand30.sh prepare
bash scripts/drivesuprim/evaluation/run_selector_expand30.sh export-pilot
bash scripts/drivesuprim/evaluation/run_selector_expand30.sh collect
bash scripts/drivesuprim/evaluation/run_selector_expand30.sh bundle
```

`prepare` = plan + fetch + init + cache. `export-pilot` экспортирует до четырёх
новых сцен. `collect` продолжает init/cache, затем экспортирует и размечает всю
новую часть; labels требуют полного экспорта. `bundle` объединяет готовую новую
часть с предыдущими 399 сценами. Повтор стадий возобновляет завершённые части с
проверкой hashes. Один процесс на выходной каталог; старые коллекции не изменяются.

Итоговые проверки:

```bash
bash scripts/drivesuprim/evaluation/run_selector_expand30.sh images-status
bash scripts/drivesuprim/evaluation/run_selector_expand30.sh report
bash scripts/drivesuprim/evaluation/run_selector_expand30.sh bundle-audit
```

## Пути и происхождение

В `/workspace/exp_v2/drivemem_memory` создаются новые каталоги:

- `selector_data_expansion_v3_30gb`: manifests, frozen selection, image receipts.
- `selector_collection_v3_30gb`: только новые train features/top16 labels.
- `selector_training_bundle_v3_30gb`: предыдущий bundle + новое дополнение.

Прежние origins `legacy`/`new` сохраняются. Дополнение имеет origin `expansion30`.
Reader, teacher/scorer config, numerical profile, vocabulary, candidate alignment,
stored raw PDM и SHA проверяются. Нельзя реконструировать score из filtered
components. В новом bundle сохраняются родительские bindings, поэтому прежние
исторические диагностики и неизменённый v2 trainer совместимы с ним.

## Последующее обучение

Сначала сравниваем 120/360/1000 эпох на старых 399 сценах через
`run_selector_scale.sh compare`. При расширении фиксируем выбранные условия
и сравниваем влияние данных отдельно от изменения длительности.

Запуск fit на новом bundle (по умолчанию 120 эпох, CPU):

```bash
bash scripts/drivesuprim/evaluation/run_selector_expand30.sh fit
```

Для заранее выбранной другой длительности:

```bash
SELECTOR_V2_EPOCHS=360 \
  bash scripts/drivesuprim/evaluation/run_selector_expand30.sh fit
```

Выход: `experiments/trajectory_selector_v3_30gb_e120` или `e360`. Обучение
использует прежние MLP и LO-session-out, без просмотра старого development или
fresh holdout. Gate и выбор длительности по OOF являются настройкой, а не
независимой оценкой. Local PDM не равен EPDMS. Отсутствие наблюдаемых ухудшений
не означает калиброванную безопасность.

Пути переопределяются `SELECTOR_EXPAND30_DATA`, `SELECTOR_EXPAND30_COLLECTION`,
`SELECTOR_EXPAND30_BUNDLE`, `SELECTOR_EXPAND30_PARENT_DATA`,
`SELECTOR_EXPAND30_PARENT_BUNDLE`, `SELECTOR_DATA_ARCHIVES`.
`SELECTOR_EXPAND30_GB` позволяет уменьшить лимит; значение >30 запрещено.
Изменение зафиксированных параметров/кода требует нового output root.
