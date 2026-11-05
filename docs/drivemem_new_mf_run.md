# Новые M/F-сессии, shard 174

5 ноября 2026 пользователь разрешил дополнительный объём до 8,5 GB.
Выбран конкретный архив 174: по последнему серверному manifest 8,427621512 GB.
Plan перепроверяет размер/hash по официальному metadata и останавливается при
превышении 8,5 GB, изменении состава четырёх сессий или недостатке сцен.
Этот предел относится к уникальному сжатому архиву; retries/protocol overhead
не являются строгим лимитом фактического сетевого трафика.

## Новый набор

`/workspace/exp_v2/drivemem_memory/mf_new4_shard174_v1`

Исключаются все сессии исходного budget-пилота, cached4, navtest и C/T исходного
inventory. Четыре новых сессии распределяются по seed 20261027 в M2/F2 до
outcomes. По 16 screening-сцен, всего 64; по 4 uniform + 4 challenge label-сцены,
всего 32 (M16/F16). Полный top16: 512 reference/candidate evaluations,
1024 raw/filtered scorer calls плюс baseline parity repeats при resume.

Старые M48/F32 остаются в своих roots. Новый пакет добавляет данные, не
заменяет старый банк и не запускает объединённое обучение автоматически.
Четыре сессии здесь обучающие/настроечные, не будущая независимая validation.
При объединении получается M64/F48 label-сцен; случайные старые labels
сохраняются как отдельная auxiliary-supervision политика для всех контролей.

## Запуск после переноса кода через Git

В host shell:

```bash
dm_new() {
  docker exec \
    -e OPENSCENE_DATA_ROOT=/workspace/exp_v2/drivemem_data \
    -w /workspace/DriveSuprim egorov_drivesuprim \
    bash scripts/drivesuprim/evaluation/run_drivemem_new_mf.sh "$@"
}

dm_new check
dm_new plan
```

Check проверяет Python tests, реальные shell wrappers со stub Python и Hydra
composition на сервере. Plan загружает только маленькие JSON manifests,
публикует roles/tokens/hashes/точный размер и проверяет Hydra identity.

После успешного plan:

```bash
dm_new prepare
dm_new label-pilot
```

ВНИМАНИЕ: `prepare` включает реальную разрешённую загрузку shard 174, extraction
нужных изображений, cache, GPU export и screen. Предыдущие четыре архива не
скачиваются. При уже имеющемся корректном archive SHA256 файл используется повторно.
Images хранятся в общем writable data root, архив — в общем camera cache.
Для пошагового запуска: `images`, `images-status`, `cache`, `cache-status`,
`export`, `screen`.

`label-pilot` завершает до восьми незавершённых сцен и показывает время CPU scorer.
Повторный pilot идёт к следующим сценам. Для остальных:

```bash
dm_new label
dm_new support-report
```

Успешный итог: status=complete, completed_scenes=32. Support показывает отдельно
uniform/challenge по M/F, baseline/oracle reserve и деградации по сессиям.
Данные не обещают заранее нужного числа редких DDC/TLC-событий.

В случае разрыва повторить незавершённую стадию. Одновременно запускать одну
копию на root. Завершённые scenes не пересчитываются, частичные labels
сохраняются после каждого действия. Hashes завершённых labels проверяются.

Нельзя изменять preparer/collector/экспорты после screen для обхода provenance.
При intentional protocol change задать новый DRIVEMEM_NEW_MF_ROOT. Полный
протокол моделей следующего этапа готовится после результатов support.
