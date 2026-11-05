# Контроль эффекта новых M/F-данных

5 ноября 2026. Сбор shard174 завершён: 32 label-сцены, без пропусков.
Новые ухудшения: M — DAC16, NC2, TTC2; F — NC5, TTC8. DDC и относительные
TLC-ухудшения отсутствуют; положительных изменений пяти safety-компонентов
тоже нет. Количества относятся к коррелированным действиям, не к независимым
эпизодам. В F новые NC/TTC-ухудшения приходятся на одну сессию.

Следующий контроль сохраняет прежний pair-utility алгоритм, признаки, target,
нулевой порог, lambda grid и процедуру macro F-CV. Добавляются только готовые
M/F-сцены. h и lambda переоцениваются прежней процедурой, не фиксируются численно.
Переобучаются direct_memory и direct_no_memory. Оба получают одинаковые labels.

Данные после объединения: M64/8 сессий, F48/6 сессий. Используются полные top16
старых и новых сцен. Старый случайный хвост не добавляется, как и в pair_utility_v1.
CV macro objective включает все F-сцены: так сохраняется процедура старого
алгоритма. Новый F содержит uniform и challenge; это явно выбранный data-control,
а не оценка качества при естественном распределении и не риск-калибровка.

Источник старых меток определяется замороженным `pair_utility_v1/identity.json`:
M top16 — memory_top16_v1, F top16 — action_transfer_v2. Новый source:
`mf_new4_shard174_v1`. Старые roots читаются без изменений.

Нет GPU, image transfer, simulator/scorer calls или новых labels.
Групповые folds F исключают целые сессии; M-запрос не получает доноров своей
сессии. Все решения обоих прежних development-наборов фиксируются до открытия
их outcomes. C/T labels не читаются.

## Запуск

После переноса изменений через Git:

```bash
dm_refit() {
  docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
    bash scripts/drivesuprim/evaluation/run_drivemem_expanded_utility.sh "$@"
}

dm_refit check && dm_refit plan
dm_refit all
```

Plan проверяет provenance и показывает counts без обучения и создания output.
All выполняет fit, фиксацию решений и lookup готовых query outcomes. CPU BLAS
ограничен четырьмя потоками. Доступны также fit, evaluate, report. Повторный all
проверяет model hashes и использует уже зафиксированные модели.

Output:
`/workspace/exp_v2/drivemem_memory/experiments/pair_utility_expanded_v1`.

В нём identity.json, fit.json, direct_memory.npz, direct_no_memory.npz,
query_decisions.json, report.json, per_scene/by_session CSV двух query-когорт.

```bash
docker exec egorov_drivesuprim cat \
  /workspace/exp_v2/drivemem_memory/experiments/pair_utility_expanded_v1/report.json
```

Сравнивать абсолютные gain/harm/coverage с baseline, парную разность memory
против no-memory и `comparison_to_original`: изменение результатов относительно
pair_utility_v1 для каждого метода. Не считать успехом только то, что память
менее вредна, чем контроль; нужен выигрыш над DriveSuprim и устойчивость по сессиям.

Оба query-набора уже просмотрены. Их повторная оценка остаётся development,
даже если модель обучалась на новых сессиях. Отчёт не доказывает пользу retrieval,
нулевой risk, continual improvement или full EPDMS gain. После этого data-control
решаем, оправдана ли новая ограниченная компонентная семья. Новые загрузки и
калибровка риска до этого решения не запускаются.

## Полученный результат

Серверный refit завершён; F-CV выбрал primary=baseline. Обученные rerankers
приведены как диагностические варианты, а не выбранная основная политика.

| Метод | Исходные 24 сцены: gain, п.п. / H | Cached4: gain, п.п. / H |
|---|---:|---:|
| Memory до расширения | +0,121335 / 1 | +0,090908 / 0 |
| Memory после расширения | −7,956934 / 3 | −8,526416 / 3 |
| No-memory после расширения | −7,956934 / 3 | −8,555463 / 3 |

Парное преимущество retrieval равно 0 и +0,029047 п.п. У memory reranker 22/24
и 16/16 вмешательств, суммарно 6 H на 40 сценах. Основная выбранная baseline
политика не вмешивается; эти шесть H относятся к диагностическому reranker.
Совпадение средних исходов методов не доказывает совпадение candidate IDs.

Контроль не поддержал объяснение «добавить эти данные — и текущая scalar ridge
начнёт переноситься». Он не доказывает бесполезность новых меток для других
моделей. Изменены данные, банк и результаты подбора h/lambda по той же процедуре;
добавленные F challenge-сцены меняют состав tuning-выборки.

Не выбирать старый удачный h/lambda по development и не называть baseline с
нулевым coverage успешным risk-controlled recall. Current scalar pair-utility
семейство на этом остановить. Следующая предусмотренная планом проверка — одна
отдельная ограниченная компонентная семья с теми же M/F и matched контролями.
Гипотеза: совместно обучать ограниченные Q/компонентные исходы и сохранять
baseline при неподдержанном вмешательстве. Постобработка старой ΔPDM через
монотонный sigmoid не исправит ranking; нужен новый обучаемый target/loss и
явно проверяемое правило выбора. Ограниченность выхода сама по себе не даёт
калибровки или гарантии безопасности.

В новых top16 M/F по-прежнему нет DDC-degradation и F TLC-degradation. Это
ограничивает проверку соответствующих learned safety-условий. Нельзя заявлять
полный многокомпонентный risk control. Дальнейший эксперимент остаётся
development, без новых image acquisitions и без открытия C/T labels.
