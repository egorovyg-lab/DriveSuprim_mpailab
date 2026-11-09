# Селектор v2 на 399 обучающих сценах

Эксперимент сравнивает прежний MLP (`control_v1`) и MLP с согласованным
рейтингом (`consistent_v2`) на одном наборе: 399 сцен из 22 поездок. Это позволяет
отделить влияние расширения данных от изменения алгоритма. Прежние результаты,
bundle и сбор данных не переписываются. Новые загрузки, метрики симулятора и GPU
для этого этапа не нужны.

## Запуск внутри контейнера

После переноса обновлений через Git:

```bash
cd /workspace/DriveSuprim
bash scripts/drivesuprim/evaluation/run_trajectory_selector_v2.sh check
bash scripts/drivesuprim/evaluation/run_trajectory_selector_v2.sh audit
nohup bash scripts/drivesuprim/evaluation/run_trajectory_selector_v2.sh all \
  > /workspace/exp_v2/trajectory_selector_v2.log 2>&1 &
tail -n 50 /workspace/exp_v2/trajectory_selector_v2.log
```

`all` выполняет аудит, 22 проверки по целым поездкам для каждой из двух моделей,
обучение итоговых моделей и диагностику на прежних 24+16 development-сценах.
Используется CPU с двумя потоками. Время зависит от CPU; прогресс печатается
каждые 40 эпох. Не запускайте два процесса в один выходной каталог одновременно.

После завершения:

```bash
bash scripts/drivesuprim/evaluation/run_trajectory_selector_v2.sh report
cat /workspace/exp_v2/drivemem_memory/experiments/trajectory_selector_v2/fit.json
```

Прерванный `all` можно повторить: завершённые folds и модели проходят проверку
контрольных сумм и используются повторно. Для отдельных этапов есть `fit` и
`historical`; второй требует готовый `fit.json`. Команда `report` требует
завершённой исторической диагностики.

## Что изменено в модели

Одинаковые входы: decoder features, геометрия траекторий, прогнозы метрик
DriveSuprim, исходный score, imitation log-probability, признаки сцены и ego.
Измеренные исходы поступают только в targets. Веса поездок одинаковые, независимо
от числа сцен; нормализация вычисляется только по обучающей части каждого fold.

Одинаковые MLP, инициализация, число зарегистрированных параметров, оптимизатор
AdamW, seed 20261108, 120 эпох, lr=.001, weight_decay=.001. Контроль вызывает
неизменённый v1 trainer. Attention на этом этапе не добавляется.

В v2 отдельный ranking output не используется: `rank = predicted_gain / .1`.
Gain относительно baseline равен нулю у baseline. Поэтому raw policy не может
выбрать альтернативу, которой сама предсказывает отрицательный прирост.
Старый ranking readout сохраняется в state dict для одинаковой инициализации
и числа параметров; на решения v2 он не влияет.

Вместо потери на идентификацию лучшего кандидата используется ожидаемая потеря
PDM: `sum(softmax(rank) * (best_PDM - candidate_PDM))`. Так ошибка в 90 п.п.
весит больше ошибки в 0.2 п.п. Итоговая потеря: `5 * expected_regret` плюс
прежние `5 * SmoothL1(gain, beta=.05) + .25 * BCE(benefit)` на альтернативах.
Regret учитывает величину ошибки; он не создаёт отдельной гарантии безопасности.

## Выбор порога и отчёты

Для каждой модели сохраняются raw и gated policy. Gate требует прогноз прироста
не менее .005 PDM (0.5 п.п.) и benefit estimate выше порога из
`{.5, .75, .9, 1.01}`. Порог 1.01 полностью отключает вмешательства.
Порог выбирается по uniform macro gain из leave-one-session-out предсказаний
при нулевом наблюдаемом ухудшении пяти safety-компонентов на uniform+challenge.
Прежние challenge-сцены сохраняют свою отметку. При равенстве выбирается больший
порог, а между моделями — `control_v1`. Если положительного допустимого gain нет,
primary становится baseline.

Выход: `/workspace/exp_v2/drivemem_memory/experiments/trajectory_selector_v2`:

- `identity.json`, `audit.json`: входы, код, runtime, состав обучающей выборки.
- `cv/`: веса нормализации и OOF-предсказания для каждого fold.
- `fit.json`, `fit_cv_per_scene.csv`: пороги, primary, обучение, raw/gated OOF.
- `control_v1.npz`, `consistent_v2.npz`: итоговые веса и нормализация, без pickle.
- `query_decisions.json`: решения обеих старых когорт, сохранённые до чтения исходов.
- `*_per_scene.csv`, `*_by_session.csv`, `report.json`: историческая диагностика,
  gain, harm, число вмешательств, session bootstrap, исключение одной поездки.

`paired_comparison` в `fit.json` и `report.json` показывает разность
`consistent_v2 - control_v1` на одинаковых сценах с bootstrap по целым поездкам.

В CSV есть `predicted_gain_pp`, `benefit_estimate`,
`negative_predicted_gain_selected`, изменения каждого компонента метрики.

OOF используется для настройки, поэтому его результат после выбора порога —
не независимая оценка. Старые 40 development-сцен уже просматривались и служат
только диагностикой. Четыре новые отложенные поездки не открываются. Для их
проверки после фиксации модели нужен отдельный этап. Benefit estimate не
калиброван, нулевой observed harm не гарантирует безопасность; local PDM
не равен EPDMS. Архивная выборка не репрезентативна для всех поездок.

`SELECTOR_V2_ROOT` меняет выход; `SELECTOR_BUNDLE_ROOT` меняет bundle;
`SELECTOR_V2_THREADS` и `SELECTOR_V2_EPOCHS` меняют параметры. После фиксации
identity изменение входов, кода, runtime или настроек требует нового каталога.
Завершённые эксперименты и файлы `failure.json` прежних протоколов не удаляются.
