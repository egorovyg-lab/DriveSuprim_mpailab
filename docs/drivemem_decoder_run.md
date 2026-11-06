# Последняя ограниченная абляция: decoder features

После повторного `candidate_ids` mismatch актуальный запуск описан в
[drivemem_decoder_replay_run.md](drivemem_decoder_replay_run.md).
Ниже сохранён прежний протокол естественного пула; его незавершённый sidecar
не смешивается с новым условным replay-экспортом.

Новые данные/метки/изображения не приобретаются. Teacher, vocab, candidate IDs,
scores, targets, lambda grid и политики выбора сохраняются. Старые замороженные
Python-модули не меняются. Этот пакет добавляет признаки, а не новую калибровку.

## Экспорт

Отдельный hook получает последний слой
`teacher.model._trajectory_offset_head.transformer_blocks[-1]`. GPU batch=1,
no_grad/eval; input pipeline тот же, что в frozen экспорте. Только M/F и уже
просмотренный development: исходный source104 + new-MF32 + cached4-development16
= 152 сцены. Остальные 32 new screening-сцены, C/T records и outcomes не нужны.

Для каждой сцены выполняются два прохода: без hook и с read-only hook.
Проверяются точное совпадение vocabulary/IDs/selected action, allclose scores
и heads (atol=1e-5, rtol=1e-6) с исходным frozen export и проходом без hook.
Каждый штатный head повторно применяется к captured latent; logits должны
совпасть с исходными. Это проверяет слой и порядок candidate features.
Любая ошибка останавливает экспорт; допуски автоматически не расширяются.

Teacher forward использует только доступные features; targets DatasetSSL не
используются для построения нового latent (как и в исходном exporter).
Метрика/scorer не вызываются, query outcomes не читаются для нового экспорта.

Sidecar: `/workspace/exp_v2/drivemem_memory/decoder_features_v1`.
В нём identity.json, export_index.json, features/<source>/<token>.npz, report.json.
Старый features/*.npz не перезаписывается. Одна сцена коммитится атомарно,
повторный export пропускает готовые проверенные файлы. CPU этап требует все 152.

## Единственное изменение признаков

К прежним общим признакам добавляются полный decoder-вектор кандидата и его
разность с baseline-вектором. Никакого PCA или нового encoder training. Retrieval
по прежнему scene embedding и geometry kernel сохраняется, чтобы одновременно
не менять поиск доноров. Векторы доступны всем четырём family: no-memory,
memory, random-compatible, shuffled outcomes. Метки, loss, normalization,
CV selection и три политики raw/component/guarded — прежние.

Прежние CV/fit/evaluate функции скопированы в отдельный runner с неизменной
логикой; тест сравнивает их AST. Ridge/predict/choice функции переиспользуются
напрямую. Источник старого component_utility_v1 проверяется по его identity;
изменение underlying алгоритма вместо feature ablation вызывает остановку.

Это additive feature ablation, а не заявление, что decoder representation
обязательно лучше или что исходный spatial mean был доказанной причиной провала.
Гиперпараметры выбираются заново только прежней F-CV процедурой. Старые V уже
просмотрены, свежего подтверждения этот эксперимент не даёт.

## Запуск после переноса кода через Git

В host shell:

```bash
dm_decoder() {
  docker exec \
    -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
    -w /workspace/DriveSuprim egorov_drivesuprim \
    bash scripts/drivesuprim/evaluation/run_drivemem_decoder.sh "$@"
}

dm_decoder check && dm_decoder plan
dm_decoder export-pilot
```

Check проверяет unit/CPU integration tests, все wrapper stages со stub Python
и Hydra composition. Реальный CUDA hook/parity проверяется именно export-pilot
на сервере: локально PyTorch/CUDA нет. Pilot экспортирует до четырёх ещё не
готовых сцен; partial report после него ожидаем.

Если pilot прошёл:

```bash
dm_decoder export && dm_decoder export-status
```

Ожидается status=complete, required_scenes=152, exported_scenes=152. GPU нужен
только на export, даже если CPU container смог выполнить check/plan. При CUDA
unavailable сначала восстановить GPU-доступ контейнера. Уже готовые sidecars
сохраняются. Перед полным export следует убедиться, что parity pilot успешен.

После полного экспорта:

```bash
dm_decoder all
```

Это CPU fit/evaluation. Доступны fit/evaluate/report. Четыре family × шесть
CV-folds + final fits, BLAS default 4 threads. Печатается family/held-out session.
При повторе готовый fit проверяется и переиспользуется. Не запускать две копии
на один output. Изменение code/inputs требует нового root, identity не редактировать.

Model output:
`/workspace/exp_v2/drivemem_memory/experiments/component_decoder_v1`.

```bash
docker exec egorov_drivesuprim cat \
  /workspace/exp_v2/drivemem_memory/experiments/component_decoder_v1/report.json
```

Оба query decision sets сохраняются до lookup outcomes. C/T labels не читаются.
Отчёт имеет ту же структуру с 13 методами, primary и парным guarded memory−control.
Сопоставлять с component_utility_v1 по каждой политике и когорте, не cherry-pick
только raw gain первой выборки. Нулевой harm при нулевом coverage не успех.

Если нет устойчивой пользы над DriveSuprim и matched controls, эта ограниченная
ветка завершается: не расширять threshold search/новые image purchases и не
переходить автоматически к CRC. Если есть полезный сигнал, сначала анализ по
сессиям и реальным memory outcomes, затем отдельное независимое подтверждение.

## Если экспорт остановился на candidate_ids

Первый полный серверный прогон сохранил 9/152 сцен и остановился на строгой
проверке candidate_ids. Исходный exporter и его hash не изменены. Проверка
могла обнаружить перестановку, замену элементов пула или другую численную
невоспроизводимость. Сам traceback эти причины не различает; менять допуски
или принимать новый набор кандидатов без разбора нельзя.

```bash
dm_decoder diagnose
```

Это отдельная диагностика первой незавершённой сцены. Она переиспользует тот
же inference/export code, временно перехватывает две parity-проверки и
останавливается до записи признаков. Запись index/identity/report отключена.
Исходные source code, sidecar identity и девять готовых файлов сохраняются.

Вывод показывает token, сравнение frozen/current и unhooked/hooked отдельно:
совпадение порядка, множества ID, baseline vocab ID, top16, missing/added IDs
и численные differences scores/heads после сопоставления по vocab ID.
Одинаковое множество ещё не означает корректность до numeric parity; разное
множество нельзя исправить простой перестановкой строк. Это не новый fit и
не ослабление условий абляции. Пришлите полный диагностический JSON прежде
чем продолжать export/обучение. Для диагностики нужна CUDA.
