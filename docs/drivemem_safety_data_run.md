# Дополнительный M/F dataset, 5 ноября 2026

Реализован пакет сбора данных из плана `drivemem_next_experiment_plan_2026_11_04.md`.
Модели следующего цикла не обучаются этим пакетом.

Новый root по умолчанию:
`/workspace/exp_v2/drivemem_memory/mf_expansion_v1`.

Источники: `budget20gb_v1`, общий inventory `pilot_v1`, список просмотренных
сессий `validation_cached4_v1`, локальные camera archives в
`/workspace/exp_v2/drivemem_data/.downloads/drivemem_camera`.
Новые image archives не скачиваются. Plan получает маленький официальный
mapping JSON, проверяет его фиксированный SHA256 и SHA256 локальных архивов.
Полная проверка больших архивов может занять время.

## Если plan вывел shortfall

Первый серверный план не смог набрать по 16 новых сцен в каждой сессии.
`selected=0` означает отказ от неполной квоты; число eligible_scenes включает
плотно соседствующие кадры. Например, 111 кадров с шагом 0,5 с дают лишь шесть
сцен при gap=10 с даже до исключения старых timestamps. Новый plan выводит
также точную `max_spaced_scenes` для каждой сессии.

Для решения о доступности без создания dataset и без повторной полной
SHA256-проверки 17 ГБ архивов:

```bash
dm_data capacity
```

Команда получает только небольшие официальные JSON manifests. Она показывает:
ёмкость локальных архивов; ёмкость по всем уже установленным metadata logs той
же M/F-сессии; отдельные дополнительные архивы с ценой; ограниченный greedy
план до 5 GB дополнительного compressed transfer. Это только расчёт, не загрузка
и не разрешение на неё. Присутствие файлов здесь проверяется по размеру;
SHA256 обязательно проверять перед extraction.

Если `impossible_even_with_more_archives` непуст, дополнительные images для
этих старых сессий при текущем inventory не восстановят исходную квоту. Тогда
нужен новый протокол состава M/F, а не снижение gap. Новые независимые M/F-сессии
и свежая validation должны планироваться раздельно. Не запускать prepare/label
до решения shortfall. Не менять вручную frozen manifests.

## Что выбирается

- Только исходные 6 M и 4 F сессии; другие роли не включаются.
- 16 новых screening-сцен в каждой сессии, всего 160; gap >=10 секунд между
  ними и всеми прежними selected timestamps той же сессии из трёх source manifests.
- При нехватке любого quota выводится shortfall, новый root не публикуется.
- Из screening pool выбирается 4 uniform + 4 challenge на сессию, всего 80:
  48 M и 32 F. Решения фиксируются до любых новых outcomes.
- Challenge: 2 teacher-head degradation и 2 repair proxies. Safety-компоненты
  циклически ротируются по session seed; одинаковые scores разрешаются по token.
  Proxy не является истинной вероятностью риска и не гарантирует сложные outcomes.
- Размечается весь top16 каждой сцены. Baseline включён, raw/filtered результаты
  сохраняются после каждого действия. Old labels не дополняются и не меняются.

Исходные roles budget-пилота авторитетны для его уже использованных M/F-сессий.
Пересечения с первоначальными C/T allocations inventory выводятся отдельным
`historical_role_conflicts`: такие сессии уже нельзя считать закрытым holdout.
Source development/C/T, cached4 и navtest сессии исключаются из расширения.

Новые timestamps из старых сессий — дополнительные обучающие данные, не новые
независимые контексты. Support report не даёт гарантий риска.

## Команды на сервере

После commit/push и pull изменений в host shell удобно определить функцию:

```bash
dm_data() {
  docker exec \
    -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
    -w /workspace/DriveSuprim egorov_drivesuprim \
    bash scripts/drivesuprim/evaluation/run_drivemem_safety_data.sh "$@"
}

dm_data check
dm_data plan
```

`check` запускает unit/integration tests, проверку реальных wrapper stages с
подставным Python и Hydra composition на сервере. `plan` создаёт manifests и
проверяет полную Hydra identity. При shortfall дальнейшие стадии не запускать;
нужен разбор доступности, а не ослабление gap или подмена сцен.

После успешного plan:

```bash
dm_data prepare
dm_data label-pilot
```

`prepare` последовательно повторно проверяет plan, извлекает нужные images
offline, строит cache 160 сцен, экспортирует teacher features и фиксирует
selection 80 label-сцен. CUDA требуется на export, batch size штатный.
`label-pilot` завершает не более 8 ещё не размеченных сцен и печатает timing.
Повторный pilot размечает следующие незавершённые сцены, не те же первые восемь.

Для полной разметки и отчёта:

```bash
dm_data label
dm_data support-report
```

`label` заканчивает оставшиеся из 80 сцен. Верхняя оценка первого полного прохода:
1280 reference/candidate evaluations / 2560 raw+filtered scorer calls. На resume
частичной сцены baseline повторяется для parity; это дополнительные scorer calls.
Завершённые сцены не пересчитываются. Scorer работает на CPU; CPU BLAS default 4.

При разрыве SSH этап можно повторить. Запускать одну копию collector на root.
Для фонового запуска уже подготовленного dataset:

```bash
docker exec -d \
  -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
  -w /workspace/DriveSuprim egorov_drivesuprim \
  bash -c 'exec bash scripts/drivesuprim/evaluation/run_drivemem_safety_data.sh label > /workspace/exp_v2/drivemem_safety_data.log 2>&1'

docker exec egorov_drivesuprim tail -n 60 /workspace/exp_v2/drivemem_safety_data.log
```

Без функции каждый этап запускается полной командой `docker exec ... bash
scripts/drivesuprim/evaluation/run_drivemem_safety_data.sh <stage>`.

## Артефакты и ограничения

`expansion_protocol.json`, `expansion_plan.json`, `identity.json`, `splits.json`,
`split_identity.json`, `image_preparation/`, `cache_index.json`, `export_index.json`,
`features/`, `vocab.npz`, `label_selection.json`, `top16_labels/*.json`,
`support_report.json`. Report содержит hashes labels, completed/missing counts,
baseline/oracle reserve по uniform/challenge и отдельную таблицу component support
по сессиям. Успех сбора означает status=complete и 80 завершённых label-сцен.

Изменение frozen sources/config/export/code после screen вызывает остановку.
Не редактировать identity для обхода ошибки. Частичный root после ошибки самого
preparer сохранять для диагностики и выбрать новый `DRIVEMEM_EXPANSION_ROOT`.
Все stages должны использовать одинаковые overrides root/data paths.

Отдельные stages: `images-offline`, `images-status`, `cache`, `cache-status`,
`export`, `screen`, `label-pilot`, `label`, `support-report`.
Пакет не обучает новый risk gate, не использует query outcomes и не открывает
calibration/test labels. Следующая реализация модели зависит от полученного support.
