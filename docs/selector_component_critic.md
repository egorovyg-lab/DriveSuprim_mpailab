# Критик с восемью компонентами PDM

Мотивация: учитель показывал снижение прогресса в восьми из девяти просмотренных
убыточных замен, но скалярный критик обещал положительный выигрыш. Новый
эксперимент обучает прогнозы компонентов и использует фиксированную формулу
для их итоговой оценки. Он проверяет гипотезу на всех training-сценах, а не
подбирает запрет под девять известных ошибок.

## Запуск внутри контейнера

После commit/push/pull:

```bash
cd /workspace/DriveSuprim
bash scripts/drivesuprim/evaluation/run_selector_component_critic.sh check &&
bash scripts/drivesuprim/evaluation/run_selector_component_critic.sh smoke &&
bash scripts/drivesuprim/evaluation/run_selector_component_critic.sh plan
```

`check` проверяет формулу, обучение, согласованность предсказаний, отсутствие
чтения validation-меток при fitting/inference, checkpoint и resume.
`smoke` проводит 20 коротких fits полного пайплайна на восьми искусственных
сценах и удаляет свои временные файлы. Реальные файлы для smoke не читаются.

`plan` проверяет замороженные bundle/контроль и совпадение формулы с raw-метриками
всех top16-кандидатов. При расхождении обучение блокируется, сохраняются
`formula_audit.json` и `formula_mismatches.csv`. Raw PDM не исправляется и не
подменяется значением из filtered-метрик. Отдельно только эту проверку можно
запустить командой `formula`.

После успешного plan:

```bash
nohup bash scripts/drivesuprim/evaluation/run_selector_component_critic.sh all \
  > /workspace/exp_v2/selector_component_critic_v1.log 2>&1 &

tail -n 60 /workspace/exp_v2/selector_component_critic_v1.log
```

Полный запуск по умолчанию: **две модели × три seed × 32 whole-session folds =
192 fold fits**, плюс шесть final fits; всего **198 обучений по 120 эпох**.
Используется CPU с двумя потоками. GPU и новые изображения не требуются.
Время зависит от сервера; это не увеличение длительности до 360/1000 эпох.

Команда `night` объединяет check, smoke и all. Для продолжения прерванного
запуска используется та же команда `all` с неизменными кодом и параметрами.
Готовые folds проверяются по hashes и не обучаются повторно. Одновременные
писатели блокируются системным lock. Статус и результат:

```bash
bash scripts/drivesuprim/evaluation/run_selector_component_critic.sh status
bash scripts/drivesuprim/evaluation/run_selector_component_critic.sh report
cat /workspace/exp_v2/drivemem_memory/experiments/component_critic_v1/summary.json
```

## Что обучается

Прежние inference-признаки: decoder vectors, геометрия траектории, teacher
heads/scores, scene/ego context. Session/token IDs и измеренные outcomes во вход
не попадают. Линейная модель и MLP шириной 64 предсказывают восемь поправок к
logits учителя; sigmoid даёт значения компонентов в [0,1]. Нулевая начальная
поправка соответствует исходным прогнозам учителя. Свободного выхода gain нет.

Порядок компонентов: NC, DAC, DDC, TLC, progress, TTC, lane keeping, history
comfort. Формула локального PDM:

```text
gate = NC * DAC * DDC * TLC
score = gate * (5*progress + 5*TTC + 2*lane_keeping + 2*history_comfort) / 14
gain(candidate) = score(candidate) - score(baseline)
```

Progress здесь уже нормализован, как в сохранённых labels. Кандидаты не
нормализуются повторно совместно. Two-frame extended comfort отсутствует.

Loss имеет прямой BCE по всем восьми компонентам, MSE их парных изменений,
MSE абсолютного score и MSE gain с повышенным штрафом за завышение. Веса
зафиксированы: component=1, delta=2, quality=5, gain=10, optimism=4,
correction penalty=.001. BCE не использует class weights, меняющие смысл
предсказаний компонентов. AdamW lr=.001, weight_decay=.01, gradient clip=1,
120 full-batch эпох. Это один заранее заданный протокол, без grid search.

Baseline тоже получает прямое обучение компонентов; его gain всегда ровно
ноль. Нормализация признаков и статистика support вычисляются только по TRAIN
fold. Поездки имеют одинаковый суммарный вес. Validation-поездка целиком
исключается из fitting и вычисления статистик.

Контроль `teacher_formula` подставляет исходные прогнозы в ту же формулу без
обучения. Сохранённые решения прежнего residual critic читаются и повторно
сверяются с raw/filtered outcomes. Неизменными остаются исходные признаки,
поездки, top16, seed, эпохи и CPU runtime. Новый алгоритм отличается и readout,
и supervision/loss; это не изолированное сравнение числа выходов.

## Ансамбль, решения и отчёт

Для ансамбля сначала усредняются восемь прогнозов компонентов, затем считается
score и gain. Это сохраняет связь gain с компонентами. Дополнительно
`gain_lower = gain - std(seed gains)` служит эвристикой согласия seed, не
доверительной границей. Линейные zero-init/full-batch реплики детерминированы;
их нулевой SD не доказывает уверенность.

Проверяются два заранее заданных режима: raw и margin 2 п.п. по gain_lower.
Baseline всегда доступен; при ties он предпочтителен. Новый вариант не
использует прежние жёсткие teacher-veto фильтры или отдельные risk heads.
Единственный кандидат на primary зафиксирован: **mlp/ensemble/gain_m020**.
Принимаются прежние критерии: positive uniform macro gain, zero observed harm
по пяти safety-компонентам, одиночная потеря не более 5 п.п., минимум десять
замен в трёх поездках и полезные улучшения минимум в трёх поездках. Иначе
primary=baseline. Linear/raw/teacher_formula остаются диагностическими контролями.

Выходной каталог:
`/workspace/exp_v2/drivemem_memory/experiments/component_critic_v1`.
В нём находятся identity, formula audit, plan/progress, cv/models checkpoints,
`*_oof_ensemble.npz` с восемью прогнозами, `oof_per_scene.csv`,
`oof_by_session.csv`, `worst_decisions.csv`, `reference_per_scene.csv`,
`report.json` и `summary.json`. В CSV решений есть предсказанные и фактические
изменения компонентов. Отчёт сравнивает ошибки компонентов с teacher,
результаты выбора и парные различия с прежним критиком на тех же сценах.

Прогнозируемый score не является гарантией безопасности или калиброванным
математическим ожиданием PDM: произведение средних компонентов в общем случае
не равно ожиданию их произведения. Улучшение отдельных head-метрик также не
заменяет проверку итогового выбора. Bootstrap по поездкам и OOF здесь служат
описанию training tuning; четыре fresh holdout остаются закрытыми.

Переменные путей: `SELECTOR_COMPONENT_ROOT`, `SELECTOR_COMPONENT_BUNDLE`,
`SELECTOR_COMPONENT_REFERENCE`. Потоки: `SELECTOR_COMPONENT_THREADS`.
Реальный запуск проверяет совпадение runtime, seed и эпох с frozen reference;
изменение протокола требует нового output root. Старые файлы экспериментов
не изменяются. Удалять manifests ради обхода проверки не требуется.
