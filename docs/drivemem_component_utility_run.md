# Ограниченная компонентная модель: один следующий эксперимент

Исходные данные: объединённые M64/8 сессий и F48/6 сессий. Источники и проверка
provenance переиспользуются из frozen expanded-utility loader. Никаких новых
images, labels, simulator calls или GPU-операций. Старые замороженные модули
не изменяются. Оба development-когорты уже просмотрены.

## Модель

Новые targets — абсолютные raw `pdm_score` и восемь компонентов. Обрезка
target в [.001,.999], logit-transform, multi-output ridge с unpenalized
intercept и sigmoid выхода. Функция потерь — squared error logits, а не
Bernoulli likelihood. Это ограниченные scores, не калиброванные вероятности,
не доверительные границы и не обещание оценки условного математического ожидания.
Это новое обучение; sigmoid старой delta-модели не применяется.

Признаки: full normalized scene embedding, нормированная ego history,
absolute candidate geometry/heads, context×geometry и score gap. Baseline
имеет собственный абсолютный прогноз, а не принудительный нуль. Набор доступных
входов общий для всех контролей. Конструкция признаков отличается от старой
relative delta regression, поэтому результат относится к новой семье целиком.

Memory features: до пяти совместимых M-эпизодов, action kernels по relative
geometry, абсолютная kernel support, агрегированные девять прошлых исходов,
их контраст относительно baseline-действия query и disagreement доноров.
h задаётся прежней M-only geometric-scale процедурой, без F/V подбора h.
Средний pooled recall с усадкой к нулю — признак, не вероятность.

Четыре отдельно обучаемых family: no_memory, memory, random-compatible,
shuffled-memory outcomes. Все получают одинаковые истинные supervised targets.
В shuffle переставляются целые девятимерные outcome vectors между action keys
внутри command groups. Своя M-сессия исключается ДО shuffle, чтобы её labels
не перешли к другим донорам и не стали собственным признаком. Истинные targets
для обучения не перемешиваются. Seed фиксирован. Это один воспроизводимый
контроль, не оценка вариативности по большому числу seeds.

## Три правила выбора на одних прогнозах

- `raw`: максимальный predicted Q; exact/numeric ties в пользу baseline.
- `component`: только действия без предсказанного снижения NC/DAC/DDC/TLC/TTC
  относительно baseline; среди них максимальный predicted Q.
- `guarded`: дополнительно ни один штатный teacher safety-head не снижается
  относительно baseline. Это строгий инженерный контроль.

Допуск 1e-8 — численный, не настроенный вероятностный margin. Baseline всегда
допустим; при отсутствии совместимых M-доноров memory families сохраняют его.
Teacher-head rule может отклонить реальные полезные действия из-за небольшого
падения head. Его цена видна в raw/component/guarded абляциях. Нулевой coverage
не считается полезной памятью. Предсказанное неухудшение не гарантирует реального.
Особенно слабо проверяются DDC/TLC, для которых поддержка labels недостаточна.

## Выбор параметров

Leave-one-F-session-out CV. Label weighting — одинаковый суммарный вес сцены;
нормировка и intercept только по training fold. M-запросы исключают всю свою
сессию из recall. F не попадает в donor bank. Для каждого family один общий
lambda из {1,10,100,1000}, затем три политики оцениваются на тех же весах модели.

Выбор lambda по guarded-policy: предпочесть нулевой observed harm на всём F,
затем uniform macro-session gain, меньше harm, сильнее regularization. Старый
F-срез классифицируется как metadata-selected uniform; новый challenge тег
берётся из frozen label_selection. Challenge gain не подменяет natural-quality
метрику. Не смешивать это с прошлым refit, где CV objective был pooled F.

Primary выбирается только среди memory_guarded/no_memory_guarded при нулевом
F-CV harm и положительном uniform macro gain; иначе baseline. В точной ничьей
предпочесть no-memory. Random/shuffle — контроли, не источники cherry-pick primary.
CV используется и для выбора lambda; его числа не являются независимым
подтверждением и не дают сертификат риска. Threshold grid и новый backbone
в этом цикле отсутствуют.

## Запуск

После переноса изменений через Git в host shell:

```bash
dm_component() {
  docker exec -w /workspace/DriveSuprim egorov_drivesuprim \
    bash scripts/drivesuprim/evaluation/run_drivemem_component_utility.sh "$@"
}

dm_component check && dm_component plan
dm_component all
```

Расчёт на CPU, default BLAS 4 threads. Он тяжелее прежнего refit: четыре
семейства × шесть CV folds плюс четыре финальных fit. Код печатает family/fold.
Между такими сообщениями идёт eigendecomposition, это не обязательно зависание.
Не запускать одновременно несколько копий на один root.

Доступны fit/evaluate/report. Повтор all использует frozen fit, проверяя hashes.
После изменения кода/inputs задавать новый DRIVEMEM_COMPONENT_ROOT; identity
не редактировать. Сохранение обеих query-decisions предшествует чтению oracle
outcomes. C/T labels не используются.

Output:
`/workspace/exp_v2/drivemem_memory/experiments/component_utility_v1`.

В нём identity.json, fit.json, четыре model.npz, fit_cv_per_scene.csv,
query_decisions.json, report.json и per_scene/by_session CSV каждой когорты.
Report содержит baseline и 12 вариантов (4 family ×3 политики), gain, harm,
coverage и paired contrast memory_guarded−no_memory_guarded. В per_scene
сохраняются predicted gain и признак наличия совместимых доноров.

```bash
docker exec egorov_drivesuprim cat \
  /workspace/exp_v2/drivemem_memory/experiments/component_utility_v1/report.json
```

Решение: есть ли полезный gain относительно DriveSuprim, дополнительная польза
реальных memory outcomes относительно no-memory/random/shuffle, приемлемый
эмпирический harm и ненулевой coverage на нескольких сессиях. Если primary
опять baseline или эффект обеспечен одним кейсом/исчезает при контролях,
не переходить автоматически к CRC или новым image acquisitions. Это один
ограниченный exploratory experiment, не подтверждение risk-controlled continual
planning или full EPDMS улучшения.

## Полученный результат и решение

Серверный эксперимент завершён; F-CV снова выбрал primary=baseline.

| Вариант | Исходный development gain, п.п. | Cached4 gain, п.п. | H исходный / cached4 |
|---|---:|---:|---:|
| memory_raw | +1,091792 | −6,329654 | 1 / 1 |
| no_memory_raw | +1,181255 | −6,092771 | 1 / 1 |
| random_raw | +1,114413 | −5,610051 | 1 / 1 |
| shuffled_raw | +0,685399 | −5,677731 | 1 / 1 |
| memory_component | −0,233908 | 0 | 0 / 0 |
| memory_guarded | −0,136687 | 0 | 0 / 0 |

Memory_guarded вмешался в 2/24 исходных сцен: обе ухудшили PDM, ни одна не
улучшилась. На cached4 он не вмешивался вовсе. Нулевой H получен при coverage
8,3% / 0%; это сохранение baseline, не полезный recall. Парное преимущество
guarded memory +0,003202 / +0,258334 п.п. объясняет преимущество над конкретным
контролем, но не является улучшением DriveSuprim: оба baseline-relative gains
неположительны. Все четыре guarded family имеют ноль improved scenes в обоих
cohorts. Raw weights выбраны по guarded F-CV, а не отдельной raw-оптимизацией.

Реальные memory outcomes не дали устойчивого преимущества над no-memory или
одним проверенным random seed. Это не доказывает бесполезность любой памяти,
но критерии продолжения данной версии не выполнены. Не выбирать memory_raw
постфактум по положительному результату только первой когорты. C/T не открывать,
не переходить автоматически к CRC или новой закупке данных.

Завершить эту ветку targets/gates. Если продолжается исходная гипотеза статьи,
остаётся одна предусмотренная ранее ограниченная feature-абляция: отдельный
экспорт frozen candidate-conditioned decoder states вместо reliance на spatial
mean embedding и handcrafted geometry interactions. Она должна сохранить
teacher/candidates/scores, существующие labels, одинаковые query-входы у
memory/no-memory и тот же заранее объявленный selection protocol. Это новая
проверяемая гипотеза, не установленное объяснение ошибки и не обещание gain.
Никакой новый privileged query input не разрешается. Если преимущество реальных
memory outcomes не появится и в ней, пересматривать постановку/claims статьи,
а не продолжать подбор thresholds на тех же семи development-сессиях.
