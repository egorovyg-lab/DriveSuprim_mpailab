# Decoder export: явный бюджет численного дрейфа

На сцене dbb6af7aa1415da4 детерминированные unhooked/hooked forward совпали
точно по всем logits и selection scores. Но один DAC-logit и два IMI-logit
вышли за старый комбинированный допуск к исходному экспорту. Максимумы:
DAC 1.526e-5, IMI 2.575e-5; selection scores 2.861e-6 остаются в допуске.
Это свидетельство для одной сцены, а не подтверждение всех 152 или другого GPU.

Новый протокол явно меняет критерий близости к старым logits. Он не называется
прежним strict parity и не перезаписывает его identity/failure. Фиксированный
предел выбран как инженерный критерий до полного прогона, по численной
диагностике, без использования результатов качества на development. Это
не универсальная граница ошибки FP32 и не калибровка риска планирования.

Условия для КАЖДОЙ сцены и ВСЕХ 256 кандидатов:

- Точный повтор всех текущих heads/scores/IDs/action/trajectory без и с hook.
- Точное совпадение старых/new candidate IDs в исходном порядке и baseline.
- Точный vocabulary и выбранная траектория; состав top16 совпадает.
- Итоговые scores: прежние atol=1e-5, rtol=1e-6.
- Каждый head-logit: абсолютная разница не более 5e-5, без относительного допуска.
- Для каждого metric head: разница sigmoid-выходов не более 1e-5.
- IMI: разница log-softmax не более 1e-4 на полном сохранённом пуле.
- Все массивы конечны. Штатные heads на captured latent проверяются отдельно
  по прежнему строгому порогу; этот критерий не расширяется.

Все прежние признаки/heads, scores, top16 для обучения и labels берутся из
старых frozen файлов. Из нового forward используется только decoder latent
в согласованном порядке IDs. Новый протокол не изменяет scorer outcomes,
старые планы, банк памяти или алгоритм обучения. Натуральный coarse pool
по-прежнему заменяется сохранённым: это условная feature ablation.

Отличия каждой сцены записываются в `export_index.json:numeric_audit`.
`report.json:numeric_audit` содержит максимумы по всем сценам и число сцен,
выходивших за прежний допуск. Поля `field_maxima` — максимумы за одну сцену,
включая счётчики `legacy_failed_elements`, а не сумма по датасету.
При любой ошибке сохраняется failure.json и повторный экспорт блокируется.
Границы не повышаются автоматически по мере прохождения новых сцен.

## Запуск после переноса кода через Git

```bash
dm_num() {
  docker exec \
    -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
    -w /workspace/DriveSuprim egorov_drivesuprim \
    bash scripts/drivesuprim/evaluation/run_drivemem_decoder_numeric.sh "$@"
}

dm_num check && dm_num plan && dm_num export-pilot
```

Первые четыре pilot records включают все три известные проблемные сцены:
dbb6af7aa1415da4, 64f4a68010155184, c5c16e45232453e3. `partial, 4/152` после
успешного первого pilot ожидаем. При повторном pilot берутся следующие
незавершённые records; завершённые файлы не пересчитываются.

После pilot без traceback:

```bash
dm_num export && dm_num export-status && dm_num all

docker exec egorov_drivesuprim cat \
  /workspace/exp_v2/drivemem_memory/experiments/component_decoder_numeric_v1/report.json
```

Новые каталоги:

- `/workspace/exp_v2/drivemem_memory/decoder_features_numeric_v1`.
- `/workspace/exp_v2/drivemem_memory/experiments/component_decoder_numeric_v1`.

Старые sidecars, identity и failure.json остаются на месте. Численный профиль
сохраняется из deterministic режима (GPU 0 по умолчанию, версии и flags в
runtime.json). Все 152 сцены проходят новый контракт отдельно. Новых downloads
или scorer calls нет. Другие roots через переменные допускаются только как
отдельные новые каталоги; identity не редактировать.

При ошибке пришлите:

```bash
docker exec egorov_drivesuprim cat \
  /workspace/exp_v2/drivemem_memory/decoder_features_numeric_v1/failure.json
```

CPU learner требует полный экспорт и проверяет hash нового контракта.
Результаты остаются development, local PDM без extended comfort, не full EPDMS.
Успешный численный аудит сам по себе не свидетельствует о пользе памяти.
