# Контролируемое дополнение M top-16 и проверка absolute/contrast

Основание — score-ceiling аудит: все 19 oracle-positive случаев из двух просмотренных наборов теоретически достижимы, поэтому гипотеза о жёстком потолке B3 на этих данных не подтвердилась. Однако на cached4 подстановка истинных absolute residuals даёт +7.059/+7.067 п.п. при beta=.5/1, а contrast — только +0.821/+0.829 против action oracle +7.246. Это не deployable результаты, но они показывают ограничение конкретного contrast target.

При beta=1 и истинных residuals `r(a)=y(a)-p(a)` contrast даёт `p_new(a)=y(a)-y(a0)+p(a0)` до clipping. Ошибка исходной оценки baseline остаётся общей ошибкой альтернатив; clipping дополнительно искажает различия. Absolute вариант в этом oracle-пробнике даёт `p_new(a)=y(a)`. Это мотивирует парный эксперимент, а не доказывает, что обучаемая абсолютная память будет работать.

Другой фактор — M coverage: размечено 15/384 действий ranks 9–16. В новом пакете дооцениваются 369 отсутствующих M top-16 действий; прежние random labels сохраняются. С baseline repeats это 834 raw/filtered scorer calls при первоначальном запуске. Никаких новых V/C/T labels, кадров, изображений или GPU export.

## Протокол

Фиксированы исходные M/F/query-сессии, модели/представления, retrieval k=5, h исходного B3 (0.0010552772409836734), beta=.5 и пул top-16.

Четыре сравниваемых варианта:

| Имя | M labels | Инъекция |
|---|---|---|
| original_M_contrast | Прежние top8+random8 | Contrast, воспроизведение B3 |
| original_M_absolute | Прежние top8+random8 | Absolute при том же h |
| completed_M_contrast | Top16 + прежний random tail | Contrast |
| completed_M_absolute | Top16 + прежний random tail | Absolute |

`original_M_absolute` не равен старому B1: прежний B1 имел другой выбранный h. Теперь h одинаков для всех четырёх, чтобы отделить влияние меток и центрирования.

Primary выбирается из этих четырёх по среднему gain на F-сессиях; ties — по меньшему H и имени. Ни h, ни beta заново не подбираются. Ridge получает новые M labels и прежние completed F labels, выбирает lambda из 1/10/100/1000 через leave-one-F-session-out, затем обучается на M+F. Старый ridge остаётся дополнительным контролем. Есть три фиксированных random seed с тем же банком/типом коррекции, что у F-selected primary.

На двух query-наборах оценки — lookup существующих top16 outcomes. **Оба набора уже просмотрены и являются development-анализом**, а не новой проверкой обобщения. Все решения фиксируются до открытия query outcomes; первичный вариант не меняется по их результатам. Основной старый B3 должен воспроизводиться точно. C/T остаются зарезервированными.

## Запуск

Новый runner `navsim/planning/script/run_drivemem_memory_completion.py`, тест `scripts/drivesuprim/evaluation/test_drivemem_memory_completion.py`; обновлены только Bash launcher. Старые model source hashes не меняются. После commit/push и pull:

```bash
docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh completion-check

docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh completion-plan
```

Первый plan ожидает `new_memory_actions: 369`, `label_stage_scorer_calls_if_run: 834`. Plan не вызывает scorer и не создаёт output root. Числа относятся к потенциальной стадии label, а не к выполненным вызовам report/fit.

```bash
docker exec -d -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim bash -c \
  'bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh completion-all >> /workspace/exp_v2/drivemem_memory_completion.log 2>&1'

docker exec egorov_drivesuprim tail -n 60 /workspace/exp_v2/drivemem_memory_completion.log
```

Последовательность: M-only доразметка → F selection/новый ridge CV → решения для обоих query-наборов → lookup и отчёты. Отдельные стадии: completion-label / completion-fit / completion-evaluate / completion-report. Нельзя запускать параллельные writers на один root.

Каждый M action оценивается отдельно с reference и fresh cache через неизменённый исходный evaluator. Raw/filtered baseline проверяется перед новым M scoring. После каждого действия сохраняется sidecar. Прерванную разметку продолжать той же командой; готовые действия не пересчитываются, baseline repeats остаются. Незавершённый ridge fit при повторе вычисляется заново; завершённая модель заморожена hashes.

```bash
docker exec -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_drivemem_budget.sh completion-report
```

Новый root по умолчанию: `/workspace/exp_v2/drivemem_memory/experiments/memory_top16_v1`. Переопределение `DRIVEMEM_COMPLETION_ROOT` сохранять во всех командах. Source — исходный budget20gb_v1; прежняя модель — `DRIVEMEM_FROZEN_MODEL_ROOT` (action_transfer_v2); второй просмотренный набор — `DRIVEMEM_VALIDATION_ROOT` (validation_cached4_v1). Изменение source/code/runtime требует нового experiment root, original identity вручную не редактировать.

Артефакты: identity.json, memory_labels/<token>.json, fit.json, ridge.npz, query_decisions.json, два *_per_scene.csv и *_by_session.csv, report.json. Отчёт содержит F-trials, выбранный primary, старый/новый ridge, safety/quality counts, actual coverage, oracle regret/reserve и paired primary−new-ridge CI. CPU threads по умолчанию 4 (`DRIVEMEM_CPU_THREADS`). GPU и новые загрузки не требуются.

Если дополнение меток и absolute-инъекция дадут полезный сигнал, следующий шаг — заранее зафиксировать метод и проверить его на новых сессиях, не выбирать удачный вариант по текущему cached4. Если убедительного сигнала нет, простое расширение такой памяти/обучение gate не обоснованы: нужен пересмотр контекстных признаков и transfer targets либо сужение гипотезы статьи. Нулевой H на малой выборке не является гарантией; метрика остаётся Local PDM без two-frame comfort.

## Полученный результат и остановка подбора текущей формулы

Completion-report выбрал по F прежний original_M_contrast. На cached4 этот вариант сохранил −0.95794 п.п.; original_M_absolute, completed_M_contrast и completed_M_absolute дали одинаковые −0.06508 п.п., ни один не улучшил сцену. Изменения устранили одну крупную quality loss, но не породили положительные вмешательства. На старом V три изменённых варианта дали +4.55236 п.п. против +4.23561 у исходного B3; прежняя концентрация эффекта не считается новым доказательством обобщения. Новый ridge дал −0.16541 на старом V и −0.87424 на cached4. Дополнительные labels сами по себе проблему не решили.

Подбор h/beta на текущих query outcomes прекращается. До пересмотра представления/targets можно сделать одну итоговую проверку возможностей уже обогащённого семейства, без новой разметки:

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  python -u scripts/drivesuprim/evaluation/audit_drivemem_completed_families.py \
  /workspace/exp_v2/drivemem_memory/experiments/memory_top16_v1
```

Скрипт сравнивает четыре клетки controlled experiment. Сначала строит proposals всех compatible single donors, prefixes/leave-one-out retrieved-5 и random-5, проверяя воспроизведение реальных full-bank choices. Только затем использует уже известный oracle top16. Печатает компактную общую таблицу, per-session границы и крупнейшие непокрытые случаи. `--all-cases` добавляет все строки. Все модели/файлы read-only; GPU, fitting, simulator calls и новые данные не нужны. Primary, выбранный по F, явно помечен; oracle-вариант не переименовывается в выбранный метод.

Если обогащённое absolute-семейство всё ещё не предлагает полезных альтернатив в достаточном числе сцен/сессий, отказаться от дальнейшего тюнинга этого residual-transfer семейства. Следующая гипотеза должна менять targets/контекстное представление, с обязательным таким же изменением сильного no-memory контроля и свежими сессиями для последующей проверки. Если capacity заметно выросла, но реальный reranker её не использует, становится обоснованной отдельная проверка выбора/агрегации опыта на M/F. Это oracle-capacity, не результат обученного selector; проверка не перебирает все комбинации пяти доноров. C/T остаются закрытыми.
