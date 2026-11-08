# Расширение данных selector: до 30 ГБ дополнительных архивов

Пользователь разрешил до 30 ГБ дополнительных изображений. Это лимит уникальных
сжатых архивов, которые ещё не скачаны. HTTP overhead в него не входит.
Скачанные архивы используются повторно бесплатно по сетевому трафику.
Они сохраняются для дальнейшего использования. Нужны свободное место для
архивов и распакованных изображений; download/extraction проверяют место.

## Цель и распределение

Сначала расширяем данные, затем возвращаемся к согласованному MLP-selector.
Количество кандидатов не заменяет количество независимых поездок.

- Исходные 112 M/F сцен и их старые файлы сохраняются.
- Целевой ориентир планирования — 16 новых сессий, из которых 4 откладываются.
- Новая сессия должна давать минимум 16 сцен с gap >=10 секунд.
- Максимум суммарно 64 обучающие сцены на сессию, включая прежние сцены.
- Дополнительные timestamps в старых M/F расширяют обучение, но не число сессий.
- Новые сессии разделяются по фиксированному seed до чтения любых outcomes.
- В каждой новой holdout-сессии сохраняются 16 сцен. Holdout не размечается
  этим скриптом и не используется при подборе модели.
- Исследованные development, исходные C/T и navtest сессии исключаются.
  Исторические конфликты ролей уже использованных M/F раскрываются отдельно.

16 новых сессий — ориентир, не обещание наличия в бюджете. План явно показывает
`target_met`, shortfall и фактическое количество. При недостатке даже для 4
holdout + 1 training session подготовка останавливается до скачивания архивов.
Поиск — детерминированная multistart greedy эвристика по metadata/стоимости,
а не доказательство оптимума. Такая выборка не является репрезентативным benchmark.
Независимость holdout от исходного pretraining DriveSuprim не установлена.

## Запуск внутри контейнера

После commit/push/pull новых файлов:

```bash
cd /workspace/DriveSuprim
bash scripts/drivesuprim/evaluation/run_selector_dataset.sh check
bash scripts/drivesuprim/evaluation/run_selector_dataset.sh plan
```

`plan` получает небольшие официальные JSON manifests, проверяет source split
checksums, учитывает полностью имеющиеся RGB сцены и cached archive coverage.
Он не скачивает изображения, не читает PDM labels, не запускает модели/scorer.
Требуется только установленный NumPy, GPU не используется.

Далее загрузка выбранных архивов и извлечение только требуемых изображений:

```bash
nohup bash scripts/drivesuprim/evaluation/run_selector_dataset.sh fetch \
  > /workspace/exp_v2/selector_dataset_fetch_v2.log 2>&1 &
tail -n 40 /workspace/exp_v2/selector_dataset_fetch_v2.log
```

Проверка результата:

```bash
bash scripts/drivesuprim/evaluation/run_selector_dataset.sh status
cat /workspace/exp_v2/drivemem_memory/selector_data_expansion_v2/capacity_report.json
```

`fetch` проверяет SHA256 архивов перед извлечением, поддерживает продолжение
загрузки `.part`, сохраняет checksummed receipts. Отсутствующий/испорченный
архив, который план считал cached, вызывает остановку до новых transfers;
бесплатный архив не превращается незаметно в дополнительную загрузку.
Совпавшие изображения сохраняются, отличающиеся существующие файлы не заменяются.
Один процесс на output root. Повторный `fetch` возобновляет работу.

## Артефакты и следующий этап

Новый root: `/workspace/exp_v2/drivemem_memory/selector_data_expansion_v2`.
`acquisition_plan.json` фиксирует код, параметры и входы;
`selection.json` — новые сцены, старые training anchors и закрытые сессии;
`image_requests.json` — past/current cameras и hashes source logs;
`capacity_report.json` — фактический прирост и дополнительный трафик;
`receipts/`, `installed_images.json`, `image_status.json` — проверенные изображения.

`complete` у status означает только готовность изображений. Обучающих labels и
новых decoder features пока нет. Это отдельный этап после проверки фактического
объёма: небольшой export/label pilot для оценки времени, затем сбор top16 labels
на train. Временной бюджет и поддержку DAC нужно измерить, не обещать заранее.
Сложные DAC/NC/TTC примеры собираются только в train; обычная часть выборки
сохраняется отдельно. Старые collectors с фиксированными квотами 32/80/152 сцен
и `trajectory_selector_v1` loader на этот новый manifest запускать нельзя.

Иные пути задаются `SELECTOR_DATA_ROOT`, `SELECTOR_DATA_ARCHIVES` и другими
переменными wrapper. Опции Python (например `--target-new-sessions`,
`--holdout-sessions`, `--max-scenes`) одинаковы на всех стадиях.
После freeze изменения кода, параметров и источников требуют нового output root.
Тесты пишут только в автоматически удаляемые временные каталоги.
