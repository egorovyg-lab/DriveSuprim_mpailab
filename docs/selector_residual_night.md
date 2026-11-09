# Ночной эксперимент: остаточный критик и пять компонентов риска

Готовые данные: 651 train-сцена, 32 поездки, top16 кандидатов и сохранённые
raw/filtered исходы. Новых архивов, GPU export и scorer calls не требуется.
Модели учатся на CPU. Этот скрипт не оценивает старые development-когорты или
четыре fresh holdout. Результат служит настройке следующего метода, а не
независимым доказательством улучшения или безопасности.

## Быстрая проверка и ночной запуск

После commit/push/pull, внутри контейнера:

```bash
cd /workspace/DriveSuprim
bash scripts/drivesuprim/evaluation/run_selector_residual_night.sh check
bash scripts/drivesuprim/evaluation/run_selector_residual_night.sh smoke
bash scripts/drivesuprim/evaluation/run_selector_residual_night.sh plan
```

`check` — небольшие unit/integration tests. `smoke` — 20 коротких обучений
полного пайплайна на восьми искусственных сценах из четырёх сессий, включая
checkpoint replay и повторное чтение завершённого эксперимента. Реальные
данные для smoke не открываются; его временные файлы удаляются.

`plan` проверяет реальные manifests, hashes, признаки, метрики и исходы
замороженного контроля на этих же 651 сценах. Показывает состав и число fit.
Создаёт identity нового эксперимента, но модели не обучает.

```bash
nohup bash scripts/drivesuprim/evaluation/run_selector_residual_night.sh night \
  > /workspace/exp_v2/selector_residual_night_v1.log 2>&1 &
tail -n 60 /workspace/exp_v2/selector_residual_night_v1.log
```

`night` сначала повторяет `check` и `smoke`, затем выполняет полный эксперимент.
Две модели, три seed, 32 leave-one-session-out folds: **192 fold fits + 6
итоговых fits = 198 обучений по 120 эпох**. Время зависит от CPU; завершение
ровно за ночь не гарантируется. Все fits идут последовательно, по умолчанию
с двумя CPU threads. Эпохи 1/40/80/120 пишутся в историю и лог. Отдельно `all`
запускает основной эксперимент без повторения синтетических тестов.

Контроль прогресса, безопасный во время обучения:

```bash
bash scripts/drivesuprim/evaluation/run_selector_residual_night.sh status
```

На завершённом эксперименте:

```bash
bash scripts/drivesuprim/evaluation/run_selector_residual_night.sh report
cat /workspace/exp_v2/drivemem_memory/experiments/residual_critic_night_v1/summary.json
```

`report` проверяет сохранённые hashes и печатает компактный отчёт.
`report --full-report` печатает также per-seed результаты, модели, histories и
полный список артефактов. Для разбора ошибок есть `worst_decisions.csv`.

## Что сравнивается

Линейная модель и MLP шириной 64 с тем же типом stem, что у прежнего селектора.
У обеих шесть выходов: поправка к teacher score и logits ухудшения NC, DAC,
DDC, TLC, TTC относительно baseline. Используются прежние inference features:
decoder vectors, geometry, teacher heads/scores, scene/ego context. Метрики
из симулятора, token и session ID не становятся входами сети.

Начальная оценка — `clip(slope * (teacher_score - baseline_score), -1, 0)`.
Неотрицательный slope обучается одномерной регрессией только на TRAIN fold и
ограничен [0,1]. Поправка альтернативы — `.1 * tanh(head)` в единицах PDM;
поправка baseline строго нулевая. Нулевая поправка воспроизводит исходный выбор
DriveSuprim, включая случаи ties. Модель не заменяет весь teacher score
произвольным рейтингом.

Loss: квадрат ошибки оценки gain с повышенным весом переоценки и действий,
ухудшающих любой safety-компонент; дополнительно пять class-weighted BCE и
штраф за величину поправки. Константы: regression weight 10, optimism weight 4,
harmful weight 4, correction penalty .1. AdamW lr=.001, weight_decay=.01,
120 full-batch epochs, gradient clipping 1. Это новый алгоритм; отличие от
прежнего MLP включает и regularization/objective, а не только число голов.

Нормализация, slope, class weights и support вычисляются только по TRAIN.
Поездки имеют одинаковый суммарный вес. Positive BCE weights ограничены 30.
Если компонент не имеет ухудшений хотя бы в двух TRAIN-поездках, risk policy
сохраняет baseline во всём соответствующем validation fold. Это ограничение
поддержки данных, а не утверждение, что редкий компонент стал безопасным.

Seeds фиксированы: 20261108, 20261109, 20261110. Лучший seed не выбирается.
Линейная модель с нулевыми весами и full-batch оптимизацией детерминирована:
её три повтора дополнительно проверяются на точное совпадение. Это проверка
воспроизводимости, а не три независимых источника информации. Нулевой seed SD
линейной модели не означает уверенность. Разные инициализации есть у MLP.
Для каждой модели используются средний gain трёх seed, `mean - SD` как
консервативная эвристика и максимальный risk estimate по seed. Mean-minus-SD
не является доверительной границей. Веса BCE меняют смысл risk estimates:
они не являются калиброванными вероятностями.

После обучения без дополнительных fits проверяются девять policies:

- `raw`: argmax среднего residual gain, baseline участвует в выборе.
- `gain_m005`, `gain_m020`: margin .005/.02 по mean-minus-SD.
- Шесть risk policies: margins .005/.02 × risk limits .05/.15/.30.

Все пять risk estimates кандидата должны пройти порог **до** argmax.
Baseline всегда допустим. При отсутствии подходящего кандидата он сохраняется.
У каждого seed те же policies оцениваются отдельно для проверки устойчивости;
настройка primary использует только ансамбли, не лучший seed.

Primary выбирается среди risk policies с положительным uniform macro gain,
нулевым observed component harm и без одиночных потерь PDM более 5 п.п.
на train OOF scenes, минимум десятью заменами в трёх поездках. Улучшения
не менее 0.5 п.п. должны встретиться хотя бы в трёх поездках. Небольшие
снижения PDM допускаются и явно показываются в отчёте. При отсутствии такого
варианта primary=baseline. Полное отключение вмешательств не считается
улучшением. Это строгие критерии настройки, не сертификат безопасности.

## Проверки и возобновление

- Целостность bundle и upstream inputs, whitelist features, raw labels,
  top16 alignment, полное покрытие замороженного контроля.
- Исключение всей поездки из каждого fold; validation targets не используются
  в inference и не участвуют в fitting stats.
- Числовая конечность loss, gradients, predictions; точный ноль baseline gain
  и граница поправки; полный OOF без повторов и пропусков.
- Train-only normalization, slope, weights/support повторно проверяются при
  чтении cached fold. CV binding содержит identity, seed и полный partition.
- Checkpoint replay итоговой модели проверяется на совпадение predictions.
- OS lock не позволяет двум процессам писать в один output. После crash ОС
  освобождает lock; удалять `.run.lock` не нужно.

Повтор команды `night` или `all` с неизменёнными условиями использует готовые
folds и итоговые модели. **Прерванный fold обучается заново**; внутри эпохи
optimizer не восстанавливается. Завершённый эксперимент только проверяется и
читается. Ошибки сохраняются в `failure.json` и `failures/`, прогресс — в
`progress.json`. Успешный повтор сохраняет историю прежней ошибки.

Изменение кода, данных, runtime, числа threads, seed или параметров требует
нового output root. Нельзя обходить hash checks удалением manifests.

Выход: `/workspace/exp_v2/drivemem_memory/experiments/residual_critic_night_v1`.
Там сохраняются `audit.json`, `plan.json`, `identity.json`, `cv/`, `models/`,
`*_oof_ensemble.npz`, `oof_per_scene.csv`, `oof_by_session.csv`,
`worst_decisions.csv`, `summary.json`, полный `report.json` и `progress.json`.

Пути: `SELECTOR_CRITIC_ROOT`, `SELECTOR_CRITIC_BUNDLE`,
`SELECTOR_CRITIC_REFERENCE`, `SELECTOR_CRITIC_ADDITION`.
Потоки: `SELECTOR_CRITIC_THREADS`. Epochs: `SELECTOR_CRITIC_EPOCHS`; контроль
должен иметь ту же длительность, runtime и число CPU threads. Для текущего
контроля используются 120 эпох и два потока. После фиксации identity условия
должны оставаться прежними; перед завершением hashes кода/входов проверяются ещё раз.
