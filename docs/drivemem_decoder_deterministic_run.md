# Decoder export с фиксированным режимом вычислений

Replay сохранил 25/152 сцен, но снова остановился на `head_imi`, теперь на
`c5c16e45232453e3`. Предыдущая диагностика другой сцены прошла; это показало,
что единичного успешного повтора недостаточно. Обучение не начиналось.

Новый exporter сохраняет пул прежних 256 ID, прежние проверки vocabulary,
selected action, траектории и всех scores/heads с `atol=1e-5`, `rtol=1e-6`.
Дополнительно два текущих forward без/с decoder-hook должны совпасть точно
по всем сохранённым heads, scores, ID, selected action и final trajectory.
Между проходами seed не сбрасывается. Неизвестная недетерминированная операция
или несовпадение с frozen export вызывает остановку, а не автоматический retry.

Перед запуском Python фиксируются `PYTHONHASHSEED=20261027`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8` и одна видимая GPU. До инициализации CUDA
включается `torch.use_deterministic_algorithms(True, warn_only=False)`,
отключаются cuDNN benchmarking/TF32, задаётся deterministic cuDNN и math-only
PyTorch SDP attention. Python/NumPy/Torch получают прежний seed. Настройки
основаны на [рекомендациях PyTorch 2.0.1](https://github.com/pytorch/pytorch/blob/v2.0.1/docs/source/notes/randomness.rst)
и [описании deterministic algorithms](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/__init__.py).

Эти настройки не гарантируют совпадения с ранее сохранёнными недетерминированными
вычислениями; это проверяется прежним порогом отдельно. Строгая повторяемость
двух текущих forward проверяется фактически, а не предполагается по flags.
Стабильность в другом окружении или между отдельными процессами этим не доказана.

Новые каталоги:

- `/workspace/exp_v2/drivemem_memory/decoder_features_deterministic_v1`.
- `/workspace/exp_v2/drivemem_memory/experiments/component_decoder_deterministic_v1`.

Прежний exporter, pool helper, 25 replay sidecars и 9 обычных sidecars остаются
неизменными. Все 152 сцены экспортируются отдельно под новым протоколом.
Дополнительной загрузки изображений или scorer calls нет. `runtime.json`
сохраняет реальные Torch/CUDA/cuDNN версии, GPU и настройки. Каждая новая запись
index связана с hash runtime и флагом подтверждённого точного повтора.
При несовпадении runtime продолжение старого нового-каталога запрещается.

## Запуск после переноса кода через Git

На сервере в host shell:

```bash
dm_det() {
  docker exec \
    -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
    -w /workspace/DriveSuprim egorov_drivesuprim \
    bash scripts/drivesuprim/evaluation/run_drivemem_decoder_deterministic.sh "$@"
}

dm_det check && dm_det plan && dm_det export-pilot
```

Pilot включает обе известные проблемные сцены (`64f4a68010155184`,
`c5c16e45232453e3`) и ещё две, всего четыре. После успешного первого pilot
ожидается `status=partial`, `exported_scenes=4`. По умолчанию используется GPU 0;
для GPU 1 передайте Docker `-e DRIVEMEM_DECODER_GPU=1`. После первых сохранений
менять выбранную GPU в этом каталоге нельзя. При восстановлении доступа CUDA
сохраняйте тот же GPU и численный протокол.

После успешного pilot без traceback:

```bash
dm_det export && dm_det export-status && dm_det all

docker exec egorov_drivesuprim cat \
  /workspace/exp_v2/drivemem_memory/experiments/component_decoder_deterministic_v1/report.json
```

CPU этап требует все 152 сцены и проверяет runtime bindings. M/F labels,
обучение ridge, CV, recall и политики выбора остаются прежними. Дополнительный
режим вычислений — техническая мера воспроизводимости, а не новая модель.
C/T по-прежнему закрыты; inspected development не становится fresh holdout.

Если возникает parity failure, exporter печатает детали и сохраняет их:

```bash
docker exec egorov_drivesuprim cat \
  /workspace/exp_v2/drivemem_memory/decoder_features_deterministic_v1/failure.json
```

В отчёте отдельно сравниваются frozen/hooked и unhooked/hooked: число значений
за допуском и величина ошибки каждого head. Повторный export того же каталога
при наличии failure.json блокируется. Не удалять failure/identity для обхода;
не повышать допуски и не переходить к fit. Если фиксированный режим стабилен,
но несовместим со старым frozen export, потребуется отдельное решение по
протоколу, прежде чем продолжать эту абляцию.

Локальные тесты не используют CUDA: они проверяют настройку API через mock,
прежние parity gates, более строгий repeat check, приоритет pilot, привязки
runtime и маршруты запуска. Реальные операции Torch 2.0.1/GTX 1080 проверяет
серверный pilot. Одновременно запускать две копии на один output нельзя.
