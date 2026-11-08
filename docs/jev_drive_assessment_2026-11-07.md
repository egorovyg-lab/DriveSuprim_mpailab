# Jev и смена исследовательского направления после DriveMem

Проверено 7 ноября 2026 года. Репозиторий jev-drive: commit
`474228e19e50b56cf7b4a983580c81c163005c67` от 21 сентября 2026 года.
Это исследовательская оценка и предлагаемый протокол, не результат запуска
Jev на NAVSIM. API, платные запросы, загрузка весов и обучение не выполнялись.

## Решение

Jev имеет смысл проверить как внешний backend семантического выбора/проверки
кандидатов поверх существующего planner. Сейчас нет основания заменять им
DriveSuprim или приписывать ему доказанное улучшение автономного вождения.
Первый эксперимент должен проверять добавочную полезность семантической
информации относительно сильных локальных controls, а не только удобство API.

## Что представляет собой Jev

Jev — модель TypeSafe класса System One. По документации разработчика она
отвечает на типизированные вопросы о state: Choice, Score, Noul, а в 1.14
также Box. Вместо свободного текста возвращаются значения и распределения;
несколько вопросов об одном state обрабатываются параллельно. Разработчик
называет метод обучения RLCD — Reinforcement Learning for Calibrated Decisions.
Полный воспроизводимый рецепт обучения из этих описаний не следует.
[Введение](https://docs.typesafe.ai/introduction),
[AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer).

Независимая работа прямо отмечает закрытость размера, архитектуры и обучающих
данных. Следовательно, «добавить архитектуру Jev в DriveSuprim и дообучить» —
не готовый технический путь. В проверенном jev-drive есть API client, но нет
весов или реализации обучения самого Jev.
[Evaluation, §4.2](https://arxiv.org/html/2609.37647v2#S4.SS2),
[API client](https://github.com/Alpha-Harper-Franklin/jev-drive/blob/main/jev_drive/api.py).

На дату проверки models page перечисляет jev-1.14.0 и jev-1.13.0. Серии
экспериментов нужно привязывать к конкретному ID, а не jev-latest. Сервис не
предоставляет customer LoRA/fine-tuning через описанный workflow. Изображения
поддерживаются, но документация относит images/Box к экспериментальному,
не готовому для production режиму. Отсутствие video input нельзя заменять
предположением, что одиночный кадр содержит скорости и историю движения.
[Models](https://docs.typesafe.ai/models), [Images](https://docs.typesafe.ai/images).

Choice confidence — функция распределения вариантов, а не вероятность
столкновения. Для n вариантов и максимальной вероятности p:
`confidence=(p-1/n)/(1-1/n)`. Она меняется с составом альтернатив. Типизация
ответа предотвращает некорректную форму, но не неверный выбор действия.
[Confidence](https://docs.typesafe.ai/confidence).

## Что реально реализовано в jev-drive

Прототип разделяет perception, semantic decision и управление. В camera probe
Qwen2.5-VL-7B строит описание изображения; Jev выбирает continue, replan,
observe или defer. Выход — запрос к host, не continuous steering/throttle.
В demo движение выполняют локальный controller и A* planner с 2D bicycle
моделью и range-limited exact geometry. Код RulesProvider получает уже
вычисленные route_blocked/progress_stalled и выбирает действие обычным if.
Это сильный обязательный baseline для этого конкретного demo.
[Providers](https://github.com/Alpha-Harper-Franklin/jev-drive/blob/main/jev_drive/providers.py),
[Episode](https://github.com/Alpha-Harper-Franklin/jev-drive/blob/main/jev_drive/episode.py).

Host gate проверяет возраст observation, совпадение observation ID,
доступность кандидата и отдельный local feasibility veto. Gate не является
реализацией perception или сертификатом safety. Синхронный 2D demo при API
вызове не продвигает simulated time; он не измеряет поведение реального
асинхронного автомобиля при задержке.
[Gate](https://github.com/Alpha-Harper-Franklin/jev-drive/blob/main/jev_drive/gate.py),
[README](https://github.com/Alpha-Harper-Franklin/jev-drive).

Published probe: 27 приватных записанных изображений из девяти source groups,
два порядка вариантов. Это 54 попытки, не 54 независимые дорожные сцены.
На RTX 4090 медиана полного пути image→Qwen choice — 275 ms;
image→caption→Qwen choice — 2848 ms; image→caption→Jev — 3966 ms, причём
47/54 Jev попыток валидны. Ground-truth outcomes отсутствуют, replan/defer
в валидных Jev ответах не выбраны; closed-loop CARLA/BeamNG результатов нет.
Точные частные inputs не опубликованы, полная независимая репликация этих
чисел из публичного repo невозможна. Это измерение интерфейса/latency,
а не эффективности или безопасности вождения.
[First probe](https://github.com/Alpha-Harper-Franklin/jev-drive/blob/main/docs/first-probe.md).

Есть также отдельный Three.js JevPilot demo: локально генерируются кандидаты,
передаются таблицы доступных путей/traffic/road boundaries, а geometry,
route search и safety brake остаются в коде. Это полезный пример интерфейса
candidate selection, но privileged simulator state нельзя приравнивать к
оценке camera-only policy.
[JevPilot](https://github.com/standardagents/jevpilot).

## Почему направление может быть полезно после наших результатов

Наши эксперименты показывают oracle reserve, но не устойчивый выигрыш memory.
Decoder memory_guarded получил +4.065 п.п. на первой development когорте,
полностью за счёт одной известной DAC-сцены за вычетом одного progress-loss;
на второй не вмешивался. Это не означает, что дополнительная семантика
бесполезна: pretrained semantic model может внести информацию другого рода.
Но одна замена ridge на Jev поверх тех же бедных числовых признаков не
гарантирует нового знания о дорожной ситуации.

Рабочая гипотеза: семантическая проверка сцены помогает различать траектории,
которые похожи для learned scorer, при достаточном качестве и актуальности
наблюдений. Гипотеза относится к информации и качеству решений, а не к
маркетинговой «быстроте» конкретного API.

## Два разных сценария применения

1. **NAVSIM candidate selector.** DriveSuprim и его top16 фиксируются; семантический
   backend выбирает ID траектории или сохраняет baseline. Это можно исследовать
   на имеющихся входах и labels с минимальными изменениями evaluator.
2. **Closed-loop recovery supervisor.** Backend запрашивает наблюдение, изменение
   плана/режима или передачу управления host. Для этого нужны реализованные
   действия, текущие observations, повторные decisions, moving simulation time,
   задержки, fallback и длительные эпизоды. Наш текущий frozen-pool local PDM
   протокол сам по себе этого не проверяет. Это более крупный отдельный проект.

Рекомендация — сначала первый сценарий; второй только после полезного сигнала
и ресурсного решения по simulator/GPU. Нельзя назвать offline reranking
доказательством recovery success в CARLA.

## Предлагаемая схема без episodic memory

```
Разрешённые камеры/ego history ──> DriveSuprim ──> baseline + top16
             │                                      │
             └──> семантические наблюдения ───────────┤
                                                    v
                                      bounded decision backend
                                                    │
                                                    v
                              проверки данных/актуальности/кинематики
                                                    │
                                                    v
                            выбранная существующая траектория или baseline
```

Jev — один из backends, рядом с rules, локальным classifier и VLM. Код считает
геометрию, скорость, curvature и прочие точные величины. Модель оценивает
наблюдаемые смысловые признаки: препятствие, неоднозначная граница, перекрытие,
недостаточность наблюдений. Не следует просить её пересчитать PDM по raw XY.
[Known limitations](https://docs.typesafe.ai/model-jaggedness).

В state нужны ego status, driving command, история наблюдений и provenance;
описания кандидатов должны быть связаны с конкретными траекториями. Общее
«на дороге есть автомобиль» не показывает, какая из 16 траекторий конфликтует
с ним. Для automatic perception потребуются проверенные tracks/masks или
геометрически корректные визуализации candidates, а не только красивый caption.

Числовые predictions DriveSuprim помечаются именно predictions. AgentInput
в нашем коде содержит текущие/прошлые ego/sensors; отдельный Scene содержит
privileged annotations/maps/future. Metric cache, oracle labels, future actors,
восстановленные истинные DAC/TTC и ID плохих сцен не должны попадать в state.
Любая дополнительная привилегия требует отдельного явно обозначенного опыта
и одинакового доступа всех controls.

## Последовательность работ и остановки

### 1. Зафиксировать decision contract

Выход NAVSIM прототипа — один ID из неизменного top16 либо retain-baseline.
Choice поддерживает до 255 вариантов: весь наш пул 256 в один Choice не помещается.
Top16 уже имеет оценённый резерв; особенно на cached4 переход от top8 к top16
сильно увеличивал reserve. Поэтому сокращать его до четырёх произвольных
манёвров до проверки reserve не стоит.
[Choice](https://docs.typesafe.ai/primitives/choice).

Описание параметров candidates формируется детерминированно без outcomes.
Одинаковые state/schema/candidates предлагаются всем decision backends. Точный
model ID, prompts, порядок кандидатов, входы, версии и ответы сохраняются.
Ошибки API/таймауты приводят к предписанному fallback, учитываются в utility и
стоимости, не выкидываются из выборки и не заменяются успешными retries.

### 2. Дешёвая проверка механизма на существующих данных

Сначала 8–12 примеров для проверки формата, привязки траекторий к наблюдениям и
чувствительности к перестановке. Затем ВСЕ 40 уже просмотренных development
сцен из семи сессий — диагностическая серия, без нового заявления о held-out
performance. Не ограничиваться уже известным f27a618a5c7458ee.

Чтобы не начинать с большой VLM и дорогой интеграции, можно сначала использовать
ручные семантические observations, составленные по разрешённым current/history
images без просмотра future/outcome. Это perception upper-bound experiment,
не deployable perception и не итоговая end-to-end модель. Scene IDs и outcomes
скрываются от annotator на этапе разметки. Общая схема observations фиксируется
до получения Jev decisions; очевидные rule-based случаи остаются controls.

Сравнить DriveSuprim; простые geometry/head rules; подходящий локальный classifier
при наличии training labels; локальную language-model decision на том же state;
Jev. Правило не должно получать route_blocked как готовый target, если другим
методам предлагается угадать это по captions. Если смысловую разметку уже может
использовать простая проверка, это преимущество восприятия, не Jev.

Проверить value added относительно controls, coverage, component regressions,
progress loss, regret к oracle того же candidate set, результаты по сессиям
и leave-one-session-out; считать качество по всем сценам с fallback.
Проверить несколько заранее заданных порядков вариантов и небольшие изменения
формулировки, не выбирать лучший prompt по этим development outcomes.

**Остановиться**, если выигрывает только известная единичная сцена, более
простой backend не хуже, кандидатный смысл не заземлён или результаты сильно
зависят от порядка/стиля описания. Это новый механизм, а не обход прежнего
F-CV правила с выбором победителя по уже просмотренному development.

### 3. Automatic perception и честные controls

Только после положительного результата с проверенными observations добавить
автоматическое perception. Сравнить shared-caption→Jev и shared-caption→local
model с абсолютно одинаковым caption; отдельно direct-image VLM и, опционально,
экспериментальный direct-image Jev. Caption-only и image models получают разную
информацию — это отдельный фактор, а не прямой вывод о превосходстве модели.

Новые независимые sessions выбираются без outcomes, настройки замораживаются,
старые C/T пока не открываются для подбора. Финальный test используется один
раз после фиксации протокола. Старые 40 development сцен не становятся fresh
holdout после смены модели. Claim о пользе требует повторяемого эффекта за
пределами одной сессии, с сопоставимыми sensing/intervention budgets.

### 4. Closed loop и latency, если предыдущие этапы оправданы

Измерять полный путь perception→serialization→queue/network→decision→gate,
p50/p95/p99 и missed deadlines. В simulator time должен идти во время запроса.
При истечении deadline local policy продолжает работу; это не обещание её
безопасности, baseline failures учитываются. Нельзя увеличивать max_age до
нескольких секунд только ради принятия медленного ответа.

В repo gate требует exact current observation ID и max_age=0.5 s по умолчанию.
Если host обновляет ID быстрее cloud-ответа, решения будут отвергаться как
superseded. Нужно определить корректный жизненный цикл запроса и повторную
проверку актуальности, а не убрать gate. При 50 km/h за 4 s машина проходит
примерно 56 m: опубликованный caption pipeline нельзя считать low-level loop.

Для recovery измерять завершение маршрута вместе со столкновениями/дистанцией,
timeouts, комфортом, временем восстановления, ненужными вмешательствами и долей
движения под fallback. Нулевая аварийность за счёт постоянной остановки не успех.

## Ресурсы и воспроизводимость

API decision не требует локальной GPU для самого Jev. Published vision probe
использует Qwen2.5-VL-7B в BF16 на 4090; pyproject vision extra требует Torch>=2.5,
Transformers 5, Python>=3.10. Наш frozen container использует Python 3.9,
Torch 2.0.1+cu118 и две GTX1080 по 8 GB. Веса 7B в 16-bit сами требуют порядка
14 GB, ещё без runtime/vision/activations. Две карты по 8 GB не дают автоматически
один общий VRAM pool. Квантизация/offload — отдельные проверяемые режимы.
Основной контейнер не обновлять; использовать отдельное окружение/адаптер.
[Project requirements](https://github.com/Alpha-Harper-Franklin/jev-drive/blob/main/pyproject.toml),
[Vision probe](https://github.com/Alpha-Harper-Franklin/jev-drive/blob/main/experiments/vision_probe.py).

У Jev proprietary weights и remote service — воспроизводимость слабее полностью
локальной frozen модели. Нужны version pinning, content-addressed request cache,
raw responses, полный учёт ошибок и явная фиксация порядка запусков. До API
серии потребуются пользовательский ключ и заданный бюджет; в этой оценке ни
ключ, ни external data transfer не использовались.

## Научная перспектива

Возможный вопрос статьи: «Когда семантическая проверка действительно улучшает
motion planning при ошибочном восприятии и задержках решений?» Jev является
одним backend. Замена вызова LLM на Jev сама по себе не новизна. Adaptive reasoning
и language-action driving уже исследуются, например в
[AutoVLA](https://arxiv.org/abs/2506.13757) и [SimLingo](https://arxiv.org/abs/2503.09594).

Независимая оценка Jev на language tasks обнаруживает хорошие single-decision
результаты, но также проблемы в последовательных играх и повторение бесполезных
действий. Эти результаты не являются driving benchmark. Перенос generic
calibration на collision outcomes нельзя предполагать.
[Independent evaluation](https://arxiv.org/abs/2609.37647).

Первый следующий шаг: подготовить decision-state schema и маленький bounded
semantic-selection probe на уже имеющихся сценах, без episodic bank, без
переобучения DriveSuprim и без нового скачивания сотен гигабайт. Дальнейшая
интеграция оправдана только после измеримого преимущества над простыми controls.
