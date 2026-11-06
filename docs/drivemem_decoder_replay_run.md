# Decoder features на сохранённом пуле кандидатов

После повторной остановки на 25/152 сценах используется отдельный
[детерминированный режим](drivemem_decoder_deterministic_run.md).
Этот replay-каталог сохраняется для аудита и не дополняется новым режимом.

Повторный обычный экспорт остановился на `candidate_ids`, хотя отдельный
diagnose ранее прошёл. Это не даёт основания считать естественный top-k
воспроизводимым. Новый протокол явно условный: decoder representation
вычисляется для прежних сохранённых 256 ID в прежнем порядке. Это эксперимент
на фиксированном candidate set, а не доказательство паритета всего естественного
coarse-to-refinement pipeline.

Временный pre-hook coarse NC-head сохраняет полный `dist_status`. Hook выхода
coarse-head подставляет сохранённые ID, их vocabulary trajectories,
соответствующие строки `dist_status` и coarse logits каждого head перед
refinement. Выбор ID не использует измеренные outcomes. Отличия естественного
coarse top-k записываются отдельно для обоих проходов в `export_index.json`;
report содержит число сцен с изменением порядка и состава естественного пула.

Два forward проходят с одинаковым сохранённым пулом: без decoder-hook и с ним.
Оба должны пройти прежнюю проверку: точные ID, selected action, vocabulary и
траектория; все final scores/heads — atol=1e-5, rtol=1e-6. Captured latent повторно
подаётся в штатные final heads для проверки соответствия. Снятие этих проверок
или расширение допусков не предусмотрено. Ошибка включает token сцены.

Исходный exporter, девять готовых sidecars в `decoder_features_v1` и все
замороженные M/F/development файлы остаются прежними. В новом протоколе заново
экспортируются все 152 сцены. Новых images, downloads или scorer calls нет.
Прежние CV, targets, ridge и политики сохранены; decoder признаки одинаково
доступны memory и matched controls. C/T остаются закрытыми. Две просмотренные
development когорты не становятся независимым подтверждением.

## Запуск после commit/push/pull

На сервере в host shell создайте функцию (в новой shell её надо определить заново):

```bash
dm_replay() {
  docker exec \
    -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
    -w /workspace/DriveSuprim egorov_drivesuprim \
    bash scripts/drivesuprim/evaluation/run_drivemem_decoder_replay.sh "$@"
}

dm_replay check && dm_replay plan && dm_replay export-pilot
```

Check запускает unit/CPU tests, проверки обеих групп wrappers со stub Python
и Hydra composition. Локально CUDA/PyTorch нет; реальный replay hook и численный
паритет проверяются серверным pilot. Pilot сохраняет до четырёх сцен; его
`status=partial` ожидаем. Любая ошибка паритета останавливает запись этой сцены.

Если pilot завершился без traceback:

```bash
dm_replay export && dm_replay export-status
```

Ожидается `status=complete`, `required_scenes=152`, `exported_scenes=152`.
Затем CPU этап:

```bash
dm_replay all

docker exec egorov_drivesuprim cat \
  /workspace/exp_v2/drivemem_memory/experiments/component_decoder_replay_v1/report.json
```

Новые каталоги по умолчанию:

- `/workspace/exp_v2/drivemem_memory/decoder_features_replay_v1` — sidecar.
- `/workspace/exp_v2/drivemem_memory/experiments/component_decoder_replay_v1` — модели и отчёт.

Уже сохранённые сцены проверяются по hashes и пропускаются при повторном запуске.
Не запускайте две копии на один output. Если задавали переменные
`DRIVEMEM_DECODER_FEATURE_ROOT`/`DRIVEMEM_DECODER_MODEL_ROOT`, убедитесь, что они
указывают на отдельные новые replay-каталоги. Identity не редактировать.
При CUDA unavailable сначала восстановить GPU-доступ контейнера: CPU check
не доказывает работоспособность GPU. При новом parity failure нужен traceback;
переходить к fit или выбирать успешный повтор как подтверждение нельзя.

После завершения сравнить обе когорты и все три политики с прежним component
experiment, включая paired memory-minus-no-memory и вред по сессиям. Нулевой
harm при нулевом coverage не свидетельствует о пользе. Это последняя ограниченная
абляция признаков, без автоматического расширения данных или поиска thresholds.

## Если replay остановился на numeric parity

Для ошибки `head_imi` на token `64f4a68010155184` добавлена отдельная диагностика:

```bash
docker exec \
  -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -e DRIVEMEM_DECODER_DIAG_TOKEN=64f4a68010155184 \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_decoder_replay.sh diagnose
```

Она делает ровно одну пару forward для указанной разрешённой сцены, без
повторных попыток. Проверяет исходную identity и весь реальный export index,
затем временно выбирает одну сцену внутри процесса. Также можно проверить
уже сохранённую сцену: её запись в настоящем index не удаляется. Экспортер,
pool helper и сохранённые sidecars не изменяются; новый root не нужен.

Три сравнения: frozen против hooked replay, unhooked против hooked replay,
frozen против unhooked replay. Отдельно выводятся полный strict parity,
совпадение baseline/траектории/top16 и ошибки каждого scores/head массива.
`failed_elements`, `max_abs_diff`, `max_tolerance_ratio` и `worst_by_tolerance`
показывают величину расхождения и допустимую ошибку конкретного кандидата.
Диагностические centered/log-softmax differences IMI не заменяют исходные
условия принятия. Ни identity/index/report, ни features не записываются.

По одному traceback нельзя установить, является ли расхождение небольшим
численным шумом или более существенным изменением. До анализа этого вывода
не продолжать fit и не повышать допуски; успешный одиночный повтор диагностики
не доказывает воспроизводимость всего экспорта.
