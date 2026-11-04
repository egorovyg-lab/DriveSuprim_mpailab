# Прямая полезность действия: следующий ограниченный эксперимент

## Что установил completed-family audit

На cached4 (16 сцен / 4 сессии) oracle по top16 даёт +7,246394 п.п.
После дополнения M oracle выбора из проверенных absolute donor families даёт
лишь +0,732674 п.п. (около 10,1% резерва), а из retrieved family — +0,088361 п.п.
Для contrast соответствующий all-family результат остаётся +0,588699 п.п.
Сцена `a5e429799b6753f5` с oracle gain +99,807953 п.п. по-прежнему не исправляется.
Это верхние оценки для конкретных проверенных семейств, а не результат обученного
селектора и не перебор всех возможных комбинаций доноров.

Новые M-метки не обеспечили перенос residual transfer. На исходном development
выигрыш по-прежнему сосредоточен в одной сессии. Дальнейший подбор h/beta,
вариантов contrast и отдельных удачных доноров прекращаем.

## Гипотеза и контроль

Меняем цель с восьми покомпонентных residuals на скаляр
`PDM(candidate) - PDM(baseline)`. Проверяем, добавляет ли retrieval информацию
сверх самой смены цели. Два метода:

- `direct_no_memory`: ridge по сцене, baseline и относительным признакам действия;
- `direct_memory`: тот же ridge плюс признаки полезности действий из M.

Оба обучаются на **одинаковых** полных top16 метках M и F. `no_memory` означает
отсутствие retrieval при расчёте признаков; он также получает M как обучающие
примеры. Это важный сильный контроль дополнительного доступа к меткам.

Общие признаки: полное нормированное visual embedding, ego history, baseline
heads/geometry, изменения geometry/heads/original score и взаимодействия
контекста с изменениями. Original score — обучаемый признак, а не фиксированный
штраф при окончательном выборе. Ridge прогнозирует непосредственно ΔPDM.

Для memory-модели берутся до пяти совместимых M-доноров штатным retrieval.
Сравнивается geometry действия относительно baseline каждой сцены. Для каждого
донора вычисляются softmax-оценка прошлой ΔPDM, абсолютная поддержка
`exp(-nearest_distance/h)` и их произведение; добавляется pooled-оценка с
усадкой к нулю. Это признаки, **не калиброванная уверенность**.
Признаки памяти вычитаются в точке query baseline. h — медиана положительных
ближайших cross-session расстояний по geometry M, без сетки и query outcomes.

При обучении на M исключаются все доноры из той же сессии. F никогда не
становится донором. Банк фиксирован; это ещё не проверка continual memory.

Метки только raw local PDM, не восстановленная формула из filtered components.
Компонентные дельты raw/filtered, quality degradation и harm отчёт сохраняет
отдельно. В определении harm используются NC, DAC, DDC, TLC, TTC; progress и
comfort видны в покомпонентных дельтах и качестве.

## Протокол выбора

1. Leave-one-F-session-out CV: M + остальные F-сессии для обучения, одна F для
   проверки. RMS-нормировка обучается заново внутри каждого fold; нет intercept
   или центрирования, поэтому прогноз baseline всегда нулевой.
2. Для каждого метода λ выбирается из 1/10/100/1000 по macro-session gain F;
   при равенстве — меньше harm, затем более сильная регуляризация.
3. Primary выбирается по pooled F-CV среди baseline и двух методов. Baseline
   побеждает при равенстве. Порог действия заранее фиксирован: predicted Δ > 0.
   CV используется и для выбора λ, поэтому его число не является независимой
   оценкой качества. Мы не заявляем risk control на основании этого правила.
4. Финальные модели обучаются на всех M/F. Их файлы и параметры фиксируются.
5. Решения для исходного development и cached4 сохраняются вместе до чтения
   любого из их oracle outcomes. Затем выполняется только lookup готовых меток.

Обе query-выборки уже многократно просмотрены. Нельзя называть ни одну новым
holdout или подтверждением обобщения. Calibration и test остаются закрытыми.

## Запуск

После переноса изменений через Git на сервере, из host shell:

```bash
docker exec \
  -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh utility-check

docker exec \
  -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh utility-plan

docker exec \
  -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh utility-all
```

GPU, новые изображения и новые вызовы scorer не нужны. CPU BLAS ограничен
четырьмя потоками; число можно задать через `-e DRIVEMEM_CPU_THREADS=...`.
Check проверяет unit/integration tests и Hydra composition. Полный реальный
прогон производится на сервере; локальные проверки без Hydra его не заменяют.

Новый output:
`/workspace/exp_v2/drivemem_memory/experiments/pair_utility_v1`.
Старые эксперименты читаются без изменения. Изменение замороженного кода,
меток, индексов или моделей приводит к остановке; identity не редактировать.

Входы по умолчанию:

- M/F: `budget20gb_v1`;
- полные M top16: `experiments/memory_top16_v1/memory_labels`;
- F top16: `experiments/action_transfer_v2/fit_labels`;
- query: `budget20gb_v1` и `validation_cached4_v1`;
- готовые результаты query: `development_oracle/scenes` обоих roots.

Доступны также `utility-fit`, `utility-evaluate`, `utility-report`.
Повторный `utility-all` проверяет и использует сохранённые модели.
После изменения собственного протокола нужен новый `DRIVEMEM_UTILITY_ROOT`.

Чтобы повторно показать отчёт:

```bash
docker exec egorov_drivesuprim cat \
  /workspace/exp_v2/drivemem_memory/experiments/pair_utility_v1/report.json
```

В output: `identity.json`, `fit.json`, два `.npz`, `query_decisions.json`,
`report.json`, `<cohort>_per_scene.csv`, `<cohort>_by_session.csv`.
Ключевой контраст: `paired_memory_minus_no_memory`, обязательно вместе с
выигрышем относительно baseline, числом улучшенных/ухудшенных сцен и harm.
Bootstrap по 3–4 сессиям — только описание этой маленькой выборки.

## Решение после результата

- Если оба метода не помогают: останавливаем эту смену цели; следующий пересмотр
  касается представления сцены/риска и разнообразия M, а не сетки h/beta.
- Если помогает только direct_no_memory: полезна смена цели, вклад retrieval
  не подтверждён. Для статьи о памяти одного такого результата недостаточно.
- Если память устойчиво лучше matched control: разобрать вклад по сессиям,
  исключить доминирование одного кадра, проверить деградации и компоненты.
  Даже положительный результат здесь остаётся exploratory.
- Перед проверкой на новых данных заморозить конкретный метод и правило выбора.
  Сначала оценить стоимость acquisition новых сессий. Не скачивать архивы
  автоматически и не использовать calibration для нового подбора модели.

Для статьи сохраняются следующие отдельные задачи: полезность retrieval сверх
matched no-memory control, selective refusal на независимой calibration,
проверка на свежих сессиях, затем full EPDMS / closed-loop и growing-memory
оценка. Текущий эксперимент проверяет только первую предпосылку.

## Полученный результат и проверка safety support

`pair_utility_v1` дал +0,121335 п.п. на исходном development и +0,090908 п.п.
на cached4. Преимущество над no-memory: +0,216157 и +0,410593 п.п.
На исходном development 98,4% суммарного парного преимущества приходится на
одну сессию; на cached4 — 92,5%. Baseline-relative gain меняет знак при
исключении отдельных сессий. Это не подтверждение устойчивого улучшения.

Замена на `31f2320ece515ff7` ухудшила DDC на 0,5 и local PDM на 48,380674 п.п.
На другой сцене модель исправила TTC и получила +35,714286 п.п. На cached4
показанные улучшения и потери связаны с progress. Coverage обеих моделей 100%.

В F-CV выбран λ=100 без памяти и λ=1000 с памятью. При одинаковом λ=100
F-CV gain памяти 1,423061, контроля 1,454276 п.п.; при λ=1000 картина иная.
Это допустимое сравнение двух отдельно настроенных методов, но ещё не изоляция
информационного вклада retrieval. Не выбирать новый λ по development.

Следующая проверка читает только существующие M/F labels, признаки, модели и
development oracle. Она считает safety failures/degradations по метрикам и
сессиям M/F, проверяет воспроизведение решений, печатает predicted/actual ΔPDM
и head deltas для safety failures и двух важных oracle-сцен. Новый gate или
модель не обучаются; гиперпараметры и threshold не подбираются.

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  python -u scripts/drivesuprim/evaluation/audit_drivemem_pair_safety.py \
  /workspace/exp_v2/drivemem_memory/experiments/pair_utility_v1
```

Редкие или отсутствующие ухудшения в M/F означают недостаточную поддержку для
обучения/проверки соответствующего safety-ограничения. Наличие предупреждения
head на просмотренном development случае само по себе не обосновывает gate.

Полученный audit подтвердил отсутствие DDC-degradation в M/F top16. В M есть
29 действий с DDC<1, но ни одно не хуже собственного baseline; в F все 480
nonbaseline top16 действий имеют DDC=1. Относительные NC-degradations есть
только в одной F-сессии (3 действия), DAC — в одной M-сессии (5 действий).
Это недостаточное покрытие для проверки многокомпонентного отказа от вмешательства.

На провальном DDC-действии head падает 0,654550→0,347223, но direct_memory
предсказывает +62,986758 п.п. при измеренных −48,380674. На выбранных действиях
средний прогноз +12,543329 / +10,170242 п.п. против измеренных +0,121335 /
+0,090908. Значит, проблема включает сильный оптимизм выбранных действий,
а не только отсутствие предупреждения teacher.

Два важных oracle repair модель с памятью оценивает отрицательно: −21,069578
при измеренных +100 п.п. на исходном development и −3,000134 при +99,807953
на cached4. Одного veto уже выбранного действия недостаточно для исправления
этого ранжирования. Порог по этим просмотренным примерам не подбирать.

Перед приобретением/разметкой новых данных проверяем уже сохранённый случайный
хвост M/F. Режим ниже печатает отдельно top16 и ranks>16, не выполняя query
inference и не читая query outcomes. Хвост может дать дополнительные обучающие
примеры, но его частота нарушений не является оценкой риска deployment top16.

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  python -u scripts/drivesuprim/evaluation/audit_drivemem_pair_safety.py \
  /workspace/exp_v2/drivemem_memory/experiments/pair_utility_v1 \
  --label-support-only
```
