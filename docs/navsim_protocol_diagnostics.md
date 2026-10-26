# Проверка соответствия протоколов оценивания

Это диагностика причин расхождения, а не новый рецепт получения 89,9.
Модель, её веса, параметры выбора траектории и существующие кэши не изменяются.

## Подтверждённые различия исходников

Исходный NAVSIM v1.1:
[scorer на commit 3e8291b](https://github.com/autonomousvision/navsim/blob/3e8291bfa89ff247231e0227778840cd0a036896/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py).

Публичный DriveSuprim:
[scorer на commit 80fe792](https://github.com/William-Yao-2000/DriveSuprim/blob/80fe792d7654a596d92e20d030d1650f6f605c02/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py).

| Определение | NAVSIM v1.1 | DriveSuprim v2 |
|---|---|---|
| Положения ego, для которых считается TTC | 0–4,0 с включительно | 0–3,1 с включительно |
| Проекции TTC от каждого положения ego | 0 / 0,3 / 0,6 / 0,9 с | те же при стандартном окне 1 с |
| Последний запрашиваемый момент препятствий | 4,9 с | 4,0 с |
| Комфорт | v1 comfort | history_comfort, другие метрики комфорта |
| Нормализация EP | raw progress с маской NC*DAC | другие маски и правила нормализации |

Горизонты TTC приведены для 40 поз с шагом 0,1 с.
Предположение о несовпадении протокола таблицы 2 и нашей адаптации ещё требует проверки.

В [grid_search_unlog_expand.py](https://github.com/William-Yao-2000/DriveSuprim/blob/80fe792d7654a596d92e20d030d1650f6f605c02/navsim/planning/script/grid_search_unlog_expand.py)
авторы используют aggregate_old_pdms:

NC * DAC * (5 * EP + 5 * TTC + 2 * history_comfort) / 12.

Это формула над уже вычисленными метриками. Скрипт работает с offline vocabulary
и не доказывает, что именно так получена таблица 2 статьи.

## Запуск на сервере

Передайте новые файлы через свой Git-репозиторий, как обычно.
Контейнер egorov_drivesuprim должен быть запущен и видеть обновлённый bind mount.

Быстрый тест CSV (без PyTorch/Ray, только чтение готовых метрик):

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/check_navsim_protocols.sh
```

Он проверяет одинаковый набор токенов, отсутствие дублей и невалидных значений,
сверяет итоговые строки со средними по сценам и пересчитывает формулу ПО КАЖДОЙ СЦЕНЕ.
Подстановка средних подметрик в формулу дала бы другой, некорректный результат.

Полный контролируемый тест TTC на всех 12146 сценах (CPU, без инференса модели):

```bash
docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/check_navsim_protocols.sh --ttc-window
```

По умолчанию используются 4 CPU worker. Для предварительной проверки можно добавить
--limit 256; это выборка, её средний результат нельзя сопоставлять с полной таблицей.
--threads 4 позволяет задать число worker явно. Полный тест может занять десятки минут;
команда должна оставаться запущенной до сообщения Protocol diagnostic complete.

Чтобы запуск не зависел от SSH-сессии, на хосте можно использовать:

```bash
nohup docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
  bash scripts/drivesuprim/evaluation/check_navsim_protocols.sh --ttc-window \
  > /auto/ws/diffusiondrive_workspace/exp_v2/diagnose_protocols.log 2>&1 < /dev/null &
```

Для других запусков задайте NAVSIM_V1_RUN и NAVSIM_V2_CSV через docker exec -e;
их значения должны быть путями внутри контейнера.

## Контроль TTC

Диагностика загружает сохранённые predictions_navsim_v1.pkl и конфигурацию исходного
v1-прогона. Проверяет хэши CSV, исходников и манифеста кэша; обычные приватные модули
v1 не меняются, поэтому существующий кэш сохраняет валидность.

На каждой сцене:

1. Симулирует пару reference/prediction один раз на исходном 5-секундном v1-кэше.
2. Воспроизводит все шесть метрик v1 и сверяет с исходным CSV с допуском 1e-8.
3. Повторно вызывает только _calculate_ttc с пределом цикла 31 вместо 40.
   Шаг времени остаётся 0,1 с. Полные массивы поз, наблюдения, reference,
   дорожная карта, EP, NC, DAC и комфорт сохраняются.
4. Сверяет короткий TTC с первым нарушением, зарегистрированным полным TTC.
   Проверяются обе траектории пары, включая reference.
5. Считает изолированное изменение PDMS как NC*DAC*5/12*(TTC_short-TTC_full).

Это НЕ полный scorer v2: меняется только TTC. Используется тот же v1-кэш,
а не другой кэш с неизвестным происхождением. При несовпадении базового
воспроизведения или неполном наборе сцен средние не публикуются.
Для pickle используется доверенный файл собственного предыдущего прогона.

## Результаты и трактовка

Внутри каталога исходного v1-прогона создаётся diagnostics_protocols/<UTC-время>/:

- csv_summary.json и per_scene_csv_comparison.csv — пересчёт формулы на метриках v2;
- ttc_summary.json и per_scene_ttc_windows.csv — результат отдельного теста TTC,
  если указан --ttc-window.

Путь на хосте начинается с
/auto/ws/diffusiondrive_workspace/exp_v2/model_ckpt/test-drivesuprim_r34-pdms-v1-teacher/2026.10.25.20.06.06/diagnostics_protocols/.

Для разбора нужны строки Old formula on v2 EP/TTC/HC, таблица variant / TTC / PDMS,
Isolated TTC-window PDMS change и количество изменившихся сцен.

Если сокращение TTC даёт заметный прирост, оно объясняет измеренную часть разницы
между двумя определениями. Это не доказательство протокола авторов.
Если пересчёт CSV близок к 89,9, это основание уточнить происхождение таблицы 2;
совпадения округлённого числа недостаточно.

CSV v2 относится к другому выбору траекторий и использует EP/TTC/HC v2.
Поэтому разницу двух CSV нельзя целиком приписывать одному TTC или одному комфорту.
Не следует подбирать коэффициенты или окна по целевому числу статьи.
