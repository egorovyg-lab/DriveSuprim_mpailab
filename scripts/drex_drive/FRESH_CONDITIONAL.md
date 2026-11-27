# Свежая проверка A/C: `corridor-preview-lead-brake-gap-conditional-fresh-v1`

**Статус: проект протокола; свежий прогон не разрешён и не проведён.** `attempt-011` дал C−B +0,766 п.п. raw PDM и +2,145 п.п. прогресса на разобранном TRAIN. Архивное сопоставление C−A показало потерю прогресса −4,114 п.п.; положительный raw PDM менял знак без целевой поездки. Поэтому первичный контроль — **A**, а не B. Старый fresh-v1 A/B остаётся отдельным протоколом.

## Гипотеза и решение

- **A** — новый инференс исходного `corridor-preview`. **C** отличается только наличием `minimum_lead_gap_if_braking_m`: выбран текущий радарный лидер, и скорость сближения строго положительна до округления текущего радара. Ноль, расхождение и отсутствие лидера выключают поле; повреждённый вход останавливает запуск. При включении используется прежний прогноз для 2 м/с². Нельзя менять правило после просмотра. B не запускается.
- A и C используют одни исходные наблюдения, 40 поз за 4 с (8 решений), одну загрузку Drex Max v1 на GPU, те же веса, argmax, seed `20261124` перед каждым плечом на каждой сцене. Контрфактическая проверка запроса выполняется в причинном состоянии своего плеча, без усечения токенов. Astra использует один и тот же проверенный кеш на пару и локальный нереактивный raw PDM без human penalty filter, не официальную формулу NAVSIM v1 и не EPDMS.
- Проектный размер — **ровно 120 независимых, новых и подходящих поездок**, все подходящие сцены из каждой. Рамка и исключения фиксируются до исходов. Для выбора используются первые 120 session ID в порядке `(SHA256(ASCII("20261125:" + session_id)), session_id)`, затем выбранные ID и токены канонически сортируются. Нельзя менять рамку, seed или делать повторный отбор, если выбранный набор выходит за лимит сцен.
- Основная оценка — средняя парная **C−A по сценам**. Одни и те же выборки целых поездок для двух метрик: 20 000 реплик `Random(20261124).choices`, в каждой делим сумму дельт на суммарное число сцен. Нижняя односторонняя 95%-ная граница — 1000-е значение в отсортированном списке. Обязательны **raw PDM > 0**, **`ego_progress` > −2 п.п.** по нижним границам и ноль падений любой из семи защищённых компонент сверх `1e-8` на каждом токене. При полном наборе защищённая регрессия имеет приоритет и означает `rejected`; без неё постоянная средняя дельта по всем поездкам хотя бы одного endpoint даёт `inconclusive`. Остальные непройденные критерии на полном наборе дают `rejected`.
- При нуле событий ~2,47% — условная верхняя биномиальная граница для доли поездок с регрессией на уровне эксперимента при независимых поездках с общей вероятностью и корректном отборе. **Не** оценка мощности PDM и **не** гарантия безопасности. До регистрации утвердить доступность 120 поездок, план точности/мощности и бюджет на основании сценариев дисперсии, без доступа к исходам C/A. При неготовности остановиться, а не уменьшать N.

## Что должен предоставить владелец данных

Сначала нужен отчёт о доступности **без строк поездок, содержимого сцен или результатов**: версия и охват источника, подтверждение независимой единицы, полнота исключений, число подходящих и неопределённых поездок, число доступных сцен для оценки ресурсов. Сценарий мощности и ресурсы утверждаются отдельно. Затем, при явном разрешении, доверенный адаптер владельца формирует перечисленные ниже входы. В checkout этот адаптер не реализован: формат и доступность источника пока неизвестны. Старая выборка `fresh-test` не подходит. Неизвестное пересечение с обучением исходных Drex/DrSu отдельно фиксируется; не объявлять независимость от их обучения без подтверждения.

| Файл | Контракт |
|---|---|
| `steward.json` | `experiment_id`, `dataset:{source_id,version,sha256}`, `frame:{frame_id,universe_sha256,adapter_sha256,filter_sha256}`. `filter_sha256` — байты `navsim_straight.py`. `exclusions` содержит `archive32`, `reserve4`, `other_prior`; каждая группа содержит `evidence_sha256`, `tokens`, `session_ids`. Исключать целые поездки, использованные для обучения метода, диагностики, выбора или тестов; прошлую выборку не включать даже частично. |
| `steward.json:eligible_scenes` | **Все** причинно подходящие строки из рамки после исключений, не только победители отбора. Поля: `token`, `session_id`, `cache_sha256`, `speed_mps`, `leaders`, `corridor_objects`, `ego_bounds`. Вычислить штатными `straight_observation`, `corridor_objects` и `ego_bounds` из начального состояния и допустимой геометрии. Не включать записанное будущее, PDM, траектории или путь кеша. Точная проверка вложенных полей — `scene_record`. |
| `approval.json` | `experiment_id`, `steward`, `independence_owner`, `reviewed_by` (строка оператора), `source_approval_ref`, `independence_approval_ref`, `exclusion_review_ref`, `preoutcome_review_ref`, `precision_decision_ref`, `capacity_decision_ref`, положительные целые `max_scenes`, `msu_wall_seconds`, `astra_wall_seconds`, SHA‑256 байтов `export_sha256`, `archive_score_sha256`, `code:{revision,scoped_diff_sha256,files}`. `files` — карта байтовых хешей путей по `CODE_FILES` в runner. |
| `cache-index.json` | Только на Astra: объект `token → относительный путь к кешу` **точно** для текущей стадии (один токен для archive smoke; полный закреплённый набор для fresh). Корень кеша передаётся отдельно. Скорер проверяет хеш кеша, пересчитывает причинный вход и сравнивает его с переданным. Загружать только доверенные оператору pickle-кеши; хеш не делает неизвестный кеш безопасным. |

Ссылки аттестации означают реальные записи, которые оператор проверяет **вне** кода. Непустая строка или хеш не доказывают ни свежесть, ни независимость, ни полноту исключений. Закрепить байтовые SHA‑256 `approval.json` и полученного `manifest.json` в отдельном доверенном журнале (кто, когда, протокол и попытка) **до** fresh-инференса; не принимать хеш из той же непроверенной поставки. В манифесте есть причинные входы: доступ к нему ограничен. Кеши и будущие треки не покидают Astra.

## Ручные стадии

Работать на одной проверенной ревизии; набор исполняемых файлов должен совпадать между ролями. Переменные ниже — явно утверждённые существующие **абсолютные** пути или отдельно закреплённые SHA‑256 байтов, не значения по умолчанию. Каждый `--out` — новая директория `attempt-NNN` под своим **уже существующим** родителем, не внутри входной директории и не её предок. Для `freeze/predict/score/validate` одинаковый basename попытки, разные родители. Перенос артефактов между ролями — только вручную по отдельно разрешённому маршруту.

1. **MSU270, внутри отдельно разрешённой GPU-аллокации — только архив.** Ориентир: 1 GPU с native BF16, 4 CPU, 96 GiB и 10 минут, один процесс через `srun`. Точный старый runtime и оригиналы `attempt-002` обязательны. Установить `DREX_PYTHON` на прежний интерпретатор Python 3.12, а `PYTHONPATH` — на закреплённый Drex upstream.

   ```bash
   export PYTHONPATH="$UPSTREAM"
   srun --ntasks=1 bash scripts/drex_drive/fresh_conditional.sh archive-predict \
     --archive "$TRAIN_ARCHIVE" --prior-attempt "$PRIOR_002" \
     --weights "$WEIGHTS" --upstream "$UPSTREAM" --revision "$REVISION" \
     --diff-sha256 "$DIFF_SHA" --out "$ARCHIVE_PRED_OUT"
   ```

2. **Astra, внутри утверждённого контейнера CPU — только архив.** `DREX_PYTHON` указывает на прежний интерпретатор скорера. Использовать прежний `TEMPLATE`, исходные `TOOLS` с `rollout_replay.py` и `NAVSIM_REPO`; не искать кеши автоматически. `ARCHIVE_PREDICTION` — разрешённо установленный `prediction.json` с соседним `pairs/`; `ARCHIVE_PRED_SHA` приходит от оператора стадии 1. `ARCHIVE_CACHE_INDEX` содержит только `f43160170a665c31`.

   ```bash
   export PYTHONPATH="$TOOLS:$NAVSIM_REPO" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
   bash scripts/drex_drive/fresh_conditional.sh archive-score \
     --prediction "$ARCHIVE_PREDICTION" --prediction-sha256 "$ARCHIVE_PRED_SHA" \
     --navsim-repo "$NAVSIM_REPO" --tools "$TOOLS" --template "$TEMPLATE" --template-sha256 "$TEMPLATE_SHA" \
     --cache-root "$ARCHIVE_CACHE_ROOT" --cache-index "$ARCHIVE_CACHE_INDEX" --cache-index-sha256 "$ARCHIVE_INDEX_SHA" \
     --out "$ARCHIVE_SCORE_OUT"
   ```

   Выход — `score.json` со статусом `smoke_verified`, только если новый A воспроизвёл сохранённый raw-вектор A из `attempt-002` (`atol=1e-8`, `rtol=1e-7`, NC/TTC точно). Это техническая проверка, не предсказание результата fresh.

3. **В разрешённой зоне данных, CPU — фиксация, до исходов.**

   ```bash
   bash scripts/drex_drive/fresh_conditional.sh freeze \
     --export "$STEWARD_EXPORT" --attestation "$APPROVAL" --attestation-sha256 "$APPROVAL_SHA" \
     --archive-score "$ARCHIVE_SCORE" --operator "$REVIEWED_BY" --revision "$REVISION" \
     --diff-sha256 "$DIFF_SHA" --out "$FREEZE_OUT"
   ```

   Закрепите байтовый хеш `manifest.json` вне каталога попытки и передайте `MANIFEST_SHA` отдельным доверенным путём. При недостатке независимых поездок, неверных исключениях, неверном архивном smoke или превышении бюджета **остановитесь**.

4. **MSU270, новая отдельно разрешённая аллокация GPU — fresh A/C.** Полный бюджет уточняется до шага 3 по числу сцен; архивный smoke не задаёт бюджет автоматически. Runner не отправляет Slurm-задание.

   ```bash
   export PYTHONPATH="$UPSTREAM"
   srun --ntasks=1 bash scripts/drex_drive/fresh_conditional.sh predict \
     --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" \
     --weights "$WEIGHTS" --upstream "$UPSTREAM" --out "$PRED_OUT"
   ```

5. **Astra, утверждённый CPU-контейнер — fresh raw scores.** Настройки как в шаге 2. Один завершённый `prediction.json` и соседний каталог `pairs/` должны быть доступны по разрешённому пути.

   ```bash
   bash scripts/drex_drive/fresh_conditional.sh score \
     --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" \
     --prediction "$PREDICTION" --prediction-sha256 "$PRED_SHA" \
     --navsim-repo "$NAVSIM_REPO" --tools "$TOOLS" --template "$TEMPLATE" --template-sha256 "$TEMPLATE_SHA" \
     --cache-root "$FRESH_CACHE_ROOT" --cache-index "$FRESH_CACHE_INDEX" --cache-index-sha256 "$FRESH_INDEX_SHA" \
     --out "$SCORE_OUT"
   ```

6. **CPU в разрешённой зоне — итог без сети, CUDA или кешей.** Проверке нужны обе полные директории `pairs/` с квитанциями. Не копировать свежий манифест, позы или парный отчёт на личную машину без отдельного разрешения; слово «локально» здесь означает выполнение без подключения к другому хосту.

   ```bash
   bash scripts/drex_drive/fresh_conditional.sh validate \
     --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" \
     --prediction "$PREDICTION" --prediction-sha256 "$PRED_SHA" \
     --scores "$SCORES" --scores-sha256 "$SCORES_SHA" --out "$VALIDATE_OUT"
   ```

Коды: `0` — стадия завершена; только `validate` даёт итог `passed`, `3` — полный `rejected`, `4` — полный, но статистически вырожденный `inconclusive`, `2` — неверные входы, неполная стадия или сбой. При сбое не удалять частичные артефакты, не переиспользовать выход, не добавлять поездки и не исключать отдельные пары. До допуска реального прогона обязателен независимый просмотр протокола, источника и фиксации рамки.

Синтетическая проверка: `python -B -m unittest scripts.drex_drive.test_fresh_conditional -v`. Она не открывает holdout и не заменяет проверку среды Astra или численную сверку архивного A. Перед архивным скорингом в утверждённом контейнере можно отдельно выполнить на закреплённом Python 3.9 тест импорта и парного replay: `"$DREX_PYTHON" -B -m unittest scripts.drex_drive.test_fresh_conditional.FreshConditional.test_astra_replay_imports_do_not_evaluate_union_annotations -v`. Тест не читает реальные данные или кеши; реальный скоринг остаётся отдельной проверкой.
