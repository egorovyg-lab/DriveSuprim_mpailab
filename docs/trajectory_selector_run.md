# Обучаемый выбор траектории: первый эксперимент

Это новый supervised эксперимент без retrieval. Исходный DriveSuprim, его
кандидаты, принятый functional export и прежние результаты не изменяются.
Jev API, новые скачивания, RGB export, GPU и вызовы scorer не нужны.

## Запуск внутри контейнера

Обновить репозиторий обычным git pull. Используется уже установленный PyTorch.
Команды выполняются внутри контейнера, из `/workspace/DriveSuprim`:

Локальные 10 тестов пройдены на Python 3.12 / PyTorch 2.5.1 CPU; синтаксис
проверен для Python 3.9. В серверном окружении Python 3.9 / PyTorch 2.0.1
первым шагом также выполнить `check`. На реальных серверных данных этот новый
эксперимент локально не запускался.

```bash
cd /workspace/DriveSuprim
bash scripts/drivesuprim/evaluation/run_trajectory_selector.sh check
bash scripts/drivesuprim/evaluation/run_trajectory_selector.sh audit
bash scripts/drivesuprim/evaluation/run_trajectory_selector.sh all
```

С хоста доступен тот же запуск через:

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/run_trajectory_selector.sh all
```

Для длительного запуска внутри контейнера:

```bash
nohup bash scripts/drivesuprim/evaluation/run_trajectory_selector.sh all \
  > /workspace/exp_v2/trajectory_selector_v1.log 2>&1 &
tail -n 40 /workspace/exp_v2/trajectory_selector_v1.log
```

Не запускать два процесса с одним каталогом результатов одновременно.
Можно разбить `all` на `fit`, `predict`, `evaluate`. Завершённые CV folds и
итоговые модели повторно не обучаются; их контрольные суммы проверяются.
`report` печатает итоговый отчёт.

## Данные и выходы

Источники — `experiments/pair_utility_v1`, `mf_new4_shard174_v1`,
`decoder_features_functional_v1` внутри `/workspace/exp_v2/drivemem_memory`.
Существующие loaders проверяют происхождение и контрольные суммы. Ожидаются
112 M/F scenes (64+48), 14 sessions (8+6), признаки для top16 из принятого
152-scene sidecar. Бывшая M используется как обычная обучающая выборка.
Для проверки берутся только development: прежние 24+16 сцен. C/T не открываются.

Новый каталог: `/workspace/exp_v2/drivemem_memory/experiments/trajectory_selector_v1`.

- `audit.json`, `training_support.csv`: поддержка обучающих событий.
- `identity.json`: зафиксированные входы, код, параметры и CPU runtime.
- `cv/`: возобновляемая проверка по целым сессиям, train-only normalization.
- `mlp.npz`, `attention.npz`: веса и нормализация, без pickle.
- `fit.json`, `fit_cv_per_scene.csv`: замороженные параметры и выбор primary.
- `query_decisions.json`: решения ОБЕИХ development-когорт до чтения outcomes.
- `*_per_scene.csv`, `*_by_session.csv`, `report.json`: качество и вмешательства.

## Фиксированный алгоритм

Две сети с близким числом параметров: общий MLP и один set-attention block.
Ширина 64, 4 attention heads, без embeddings номера кандидата, vocabulary ID,
token или session. Входы: decoder vectors и разности с baseline, геометрия,
predicted metrics, imitation log-probability, исходные scores, scene/ego features
и явный baseline flag. Нормализация обучается только на training folds.

Выходы: ranking logit, predicted PDM gain относительно baseline и benefit logit.
Benefit target: прирост PDM не менее 0.005 без ухудшения любого из пяти
safety-компонентов (NC, DAC, DDC, TLC, TTC). Это относительное условие, не
абсолютная безопасность. Используется сохранённый **raw PDM**, не формула из
отфильтрованных компонентов.

Loss: масса softmax на всех кандидатах в пределах 0.001 PDM от лучшего;
дополнительно `5 * SmoothL1(gain, beta=.05) + .25 * BCE(benefit)` на альтернативах.
Потери усредняются по кандидатам, сценам и сессиям. На одинаково хороших
кандидатах ranking loss равен нулю. 120 full-batch epochs, AdamW lr=.001,
weight_decay=.001, seed=20261108. Нет выбора лучшей эпохи/seed по development.

Leave-one-entire-session-out по всем 14 M/F sessions. Для каждой сети сохраняются
raw argmax и gated policy. Gate требует predicted gain >=0.005 и benefit estimate
выше порога из `{.5,.75,.9,1.01}`; 1.01 означает сохранить baseline всегда.
Порог выбирается на M/F OOF: нулевой observed harm на uniform+challenge,
затем максимальный uniform macro gain. При равенстве — больший порог.
Primary выбирается только среди gated моделей с положительным gain и без
observed OOF harm; иначе baseline. При равенстве моделей — MLP.

## Ограничения интерпретации

OOF используется для настройки порога/primary и не даёт независимой оценки
этого выбора. Обе development-когорты уже просмотрены раньше: это диагностика.
Local PDM не равен full EPDMS; benefit estimate не является калиброванной
вероятностью безопасности. Нулевые вмешательства не доказывают улучшение.
Отчёт показывает raw/gated результаты обеих моделей, harm, coverage,
session bootstrap и leave-one-session-out gains. Для подтверждения эффекта
понадобится заранее определённый независимый набор поездок.

`DRIVEMEM_SELECTOR_ROOT` меняет выход, `DRIVEMEM_SELECTOR_THREADS` — число CPU
threads, `DRIVEMEM_SELECTOR_EPOCHS` — число эпох. Любое изменение кода, входов,
runtime или параметров требует **нового выходного каталога**. Старые файлы
экспериментов не удалять и не переписывать ради возобновления.
