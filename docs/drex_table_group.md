# Несколько GPU в одном Slurm-задании

На cn29: 128 CPU, 2 000 000 MiB RAM, 8 GPU. Один работник использует 4 CPU,
96 GiB RAM и одну A100 80 GB. Для N работников новый launcher запрашивает одним
заданием N GPU, 4N CPU и 96N GiB RAM: для 8 это 32 CPU и 768 GiB RAM.
Это вписывается в опубликованные ресурсы cn29, но свободные ресурсы определяет
Slurm. При занятых семи GPU запрос 2/4/8 будет ждать освобождения достаточного
числа карт. Массив `%8` только разрешал до восьми одновременных отдельных задач;
его нельзя трактовать как гарантированное выделение восьми карт.

`run_table_group.sh` использует один GPU allocation и `srun` с N задачами,
`--gpus-per-task=1`, `--gpu-bind=single:1`, `--kill-on-bad-exit=1`.
Каждая задача проверяет одну видимую GPU, её UUID, BF16 и >=70 GiB VRAM.
Перед загрузкой модели все задачи записывают привязки и встречаются на барьере.
Повторяющийся UUID, отсутствующий rank, другая нода или старый step останавливают
запуск. `CUDA_VISIBLE_DEVICES` вручную не переназначается.

Ночной GPU allocation выполняет два srun-прохода последовательно: все mini-pilot
workers, затем все full workers. Между ними allocation сохраняется. Ошибка любого
pilot rank останавливает job до полного прогона. Каждому worker остаётся прежний
shard и собственный процесс модели; это репликация, не разделение весов по картам.

Старые `run_table.sh`, `table_pilot.py`, таблицы, выборки, метки, условия и
метод среднего трёх перестановок не изменяются. Новый адаптер сохраняет
`group_execution.json` с SHA launcher-кода и исходного scientific plan.
Старый завершённый ночной эксперимент заново вычислять не требуется.

## Сначала маленькая проверка физической привязки

На msu270:

```bash
cd ~/drivesuprim
git pull --ff-only
export DREX_RUNTIME_ROOT=/scratch/pr0201/drex_runtime
export DREX_TABLE_NAME=table_group_bind_gpu2_v1

bash scripts/drex/run_table_group.sh submit-bind 2
```

Этот job не загружает Drex, не открывает пакет данных и не запускает scorer.
Запрашивает обе GPU одним allocation, затем проверяет два разных UUID.
Он может ждать очередь, если свободна только одна GPU. Для проверки всех восьми:
новое имя `table_group_bind_gpu8_v1`, команда `submit-bind 8`.

Файлы каждого rank: `$DREX_RUNTIME_ROOT/logs/table-group-bind-JOBID-RANK.log`.
В них должны быть одинаковые `job_id`, `step_id`, `hostname`, разные `uuid` и
`visible_devices: 1`. Полные записи и их SHA сохраняются в
`runs/$DREX_TABLE_NAME/execution/JOBID-STEP-bind-LAUNCHID/`.
У каждого srun есть отдельный launch token, чтобы не принять старые rank-файлы
после перезапуска Slurm job за текущую проверку.

## Следующий большой эксперимент

После проверки привязки и выбора нового научного эксперимента:

```bash
export DREX_TABLE_COHORT=all
export DREX_TABLE_NAME=table_group_all_gpu8_v1
bash scripts/drex/run_table_group.sh submit-night 8
```

Схема: CPU checks (оба test_table*.py) → CPU plan → ONE GPU allocation
(mini-pilot → full) → CPU report. GPU job для 8 работников запросит `gres/gpu=8`,
`cpu=32`, `mem=768G`. Массивов задач в новом launcher нет.
Для 1/2/4 использовать другое имя и соответствующее число.

Отдельный mini-pilot: `submit-pilot N`; продолжить после него: `submit N` с теми же
именем/числом карт/cohort. Готовые запросы сохраняются и повторно не считаются.
Команда `submit` сама pilot не добавляет; для автоматической проверки сначала
использовать `submit-night`. Старое `table_topk_*` имя намеренно не принимается.

Оценка top64 на Astra остаётся прежней: `table_labels.py --table-run "$DREX_TABLE_NAME"`.
Для автоматического ожидания плана/отчёта — `--wait-seconds 43200 --submit-report`.
Её login-процесс требует предварительного `run_transfer.sh connect`.
Отчёт проверяет тот же frozen scientific protocol, независимо от вида allocation.

Контроль allocation:

```bash
sacct -j JOBID -X --format=JobID,State,Start,Elapsed,AllocTRES%80
bash scripts/drex/run_table_group.sh status 8
```

В `AllocTRES` одного GPU job должны быть все запрошенные карты. Rank-логи и UUID
подтверждают раздельную привязку к картам; одно значение `TIME` само по себе этого
не доказывает. Launcher не обходит очередь и не использует чужие выделенные GPU.

Документация Slurm: https://slurm.schedmd.com/srun.html и https://slurm.schedmd.com/gres.html.
