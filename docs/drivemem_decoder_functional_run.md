# Функциональная эквивалентность после полного численного аудита

Полный audit 152/152 дал нулевые нарушения current repeat, baseline, top16,
vocabulary и latent/head alignment. Прежний numeric contract прошли 150 сцен;
две нарушили границы IMI. Максимальный дрейф scores 8.5831e-6, metric sigmoid
3.0534e-6, IMI logit 1.0109e-4, IMI log-softmax 1.1568e-4.

Этот режим явно пересматривает критерий ПОСЛЕ аудита. Он не является успешным
прохождением прежнего strict/numeric контракта и не даёт калибровки риска.
Ни для одного токена нет индивидуального исключения.

Обязательные условия для каждой сцены и всех 256 кандидатов:

- Точный current repeat без/с decoder-hook для всех heads, scores, IDs,
  selected action и final trajectory. IMI здесь тоже должен совпадать точно.
- Полный vocabulary и исходный порядок 256 уникальных ID; baseline и траектория
  совпадают с frozen вариантом, состав top16 сохраняется.
- Selection scores: прежний atol=1e-5, rtol=1e-6.
- Все восемь metric heads: абсолютный logit drift <=5e-5 и sigmoid drift <=1e-5.
- IMI должен быть конечным и иметь верную форму. Его старое/new raw и normalized
  отклонение записываются, но не являются самостоятельным порогом приёмки.
- Все девять heads заново применяются к captured latent и должны совпасть с
  текущими outputs по прежнему строгому порогу. Это условие не ослабляется.

Snapshot обязательного предыдущего аудита включает identity/report/index/runtime
и hashes ВСЕХ 152 строк, включая failures. Проверяются cohort, исходные входы,
code dependencies и functional invariants каждой строки. Изменение этих файлов
блокирует продолжение. Экспорт использует тот же runtime, что и полный audit,
и на каждой сцене независимо заново проверяет новый контракт.

## После переноса изменений через Git

```bash
dm_func() {
  docker exec \
    -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
    -w /workspace/DriveSuprim egorov_drivesuprim \
    bash scripts/drivesuprim/evaluation/run_drivemem_decoder_functional.sh "$@"
}

dm_func check && dm_func plan && dm_func export-pilot
```

Pilot включает e7832ec3cbcb5fe4 и fedf56bd69af5cec (оба нарушения прошлого
контракта), а также 64f4a68010155184 и c5c16e45232453e3. Ожидается partial,
4/152 без traceback. Настройки GPU/численного профиля остаются как в audit
(по умолчанию GPU 0). Новые outputs:

- `/workspace/exp_v2/drivemem_memory/decoder_features_functional_v1`.
- `/workspace/exp_v2/drivemem_memory/experiments/component_decoder_functional_v1`.

После успешного pilot:

```bash
dm_func export && dm_func export-status && dm_func all

docker exec egorov_drivesuprim cat \
  /workspace/exp_v2/drivemem_memory/experiments/component_decoder_functional_v1/report.json
```

Все 152 сцены экспортируются отдельно, с resume по hashes; прежние sidecars,
identities и failure.json не меняются. Feature report содержит protocol и
functional contract, полный IMI drift и число выходов за старый допуск.
`max_parity_error` в index — диагностический максимум raw drift, включая IMI,
а не общий порог функциональной приёмки.

При ошибке экспорт останавливается и сохраняет failure.json в новом каталоге.
Повторный export блокируется; пороги не меняются автоматически. Нельзя запускать
две копии в один output или удалять failure/identity для обхода проверки.

CPU этап требует все 152 sidecars, проверяет runtime и snapshot предыдущего
аудита. Старые scores/heads/labels и top16 из frozen features остаются входами
learner. Из нового forward берётся только decoder latent. Функции CV/fit/evaluate,
ridge и выбор политики сохраняются. Все controls получают те же decoder features.
C/T остаются закрыты, обе evaluation когорты уже просмотрены; это development,
local PDM, не full EPDMS и не независимое подтверждение пользы памяти.

Локально проверяется контракт, provenance, неизменность learner и wrappers.
Реальный CUDA hook/forward проверяет серверный pilot. Критерий выбран после
численного аудита; это должно явно отражаться в описании эксперимента статьи.
