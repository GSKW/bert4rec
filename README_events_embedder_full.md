# README: полный пайплайн обучения эмбеддера пользователей по последовательностям игровых событий

## 1. Что мы делаем

Мы переходим от игрушечного эксперимента к **полному боевому пайплайну** обучения эмбеддера пользователей по игровым событиям на **всех доступных данных**.

Цель проекта:

1. обучить **основную модель-эмбеддер** на полном датасете;
2. построить **эмбеддинги пользователей** основной моделью;
3. построить **baseline-эмбеддинги / baseline-представления** для сравнения;
4. оценить всё по **единому протоколу**;
5. получить финальную сводную таблицу результатов.

## 2. Базовые принципы этого этапа

### 2.1. Учим на всём датасете

Под «на всём датасете» понимается:
- не делаем игрушечную подвыборку;
- не режем данные эвристически до маленького процента;
- используем весь очищенный массив событий;
- при этом **честно делаем train / valid / test split по пользователям**.

То есть:
- модель обучается на полном наборе train-пользователей;
- валидация и тест строятся на полном valid/test-пуле пользователей;
- никаких утечек между пользователями быть не должно.

### 2.2. Не схлопываем события

На этом этапе **не делаем coarse-схлопывание событий в укрупнённые классы**.

Причина:
- наша цель — именно **сильный эмбеддер**, который учится на максимально полной структуре событий;
- агрессивное укрупнение слишком рано выкидывает информацию;
- мы хотим обучать encoder на богатом событийном сигнале, а не на заранее обеднённой разметке.

Разрешено только:
- нормализовать строки;
- канонизировать JSON;
- удалять технический мусор формата;
- стабилизировать словарь токенов;
- выделять числовые поля в бакеты;
- но **не заменять разные события одним «схлопнутым» классом только ради упрощения**.

### 2.3. Пайплайн должен быть устойчив к прерываниям сервера

Сервер даёт **прерываемые ресурсы**, значит вся система должна быть построена так, чтобы:
- любой этап можно было перезапустить;
- прогресс не терялся;
- обучение продолжалось из чекпоинта;
- промежуточные датасеты не пересчитывались с нуля;
- W&B-трекинг продолжался в том же run.

### 2.4. Никакой магии из состояния ноутбука

Нельзя строить пайплайн так, чтобы он жил только в RAM.

Каждый этап обязан писать на диск:
- промежуточные таблицы;
- vocab;
- split;
- labels;
- checkpoints;
- exported embeddings;
- final metrics.

---

## 3. Что уже задаёт общий протокол

Из существующей презентации уже следует, что финальный протокол должен включать:
- два уровня оценки: **sequence quality** и **downstream**;
- sequence quality = **предсказание следующего события**;
- downstream = **retention_7d** и **retention_14d**;
- нужен **общий split по пользователям**;
- нужны **единые labels, horizons и финальная downstream-таблица**.

Также из промежуточных результатов следует:
- ветка **SimCLR** была сильна в downstream;
- ветка **BERT4Rec** чувствительна к дизайну токенизации;
- для sequence-задачи уже использовались baseline’ы **MostPopular**, **Markov1 / Bigram**, а также **GRU / LSTM**;
- для downstream уже использовался **prefix-based протокол** без утечки по времени, с префиксами `50 / 100 / 150`.

Это всё не отменяем, а переводим в единый финальный пайплайн.

---

## 4. Целевой результат проекта

На выходе должны быть следующие артефакты.

### 4.1. Основная модель

Обученный **encoder-эмбеддер** по последовательностям событий.

### 4.2. Основные эмбеддинги

Таблица пользовательских эмбеддингов:
- для train / valid / test;
- для нескольких prefix length;
- с привязкой к user_id и labels.

### 4.3. Baseline-представления

Отдельная таблица baseline-эмбеддингов или baseline-features, которые можно использовать:
- либо как сильный baseline;
- либо как дополнительную ветку для combined setup.

### 4.4. Sequence evaluation table

Таблица качества по next-event prediction:
- MostPopular;
- Bigram / Markov1;
- RNN baseline (опционально);
- основная encoder-модель.

### 4.5. Downstream evaluation table

Таблица качества по:
- `retention_7d`;
- `retention_14d`;
- для префиксов `50 / 100 / 150`.

### 4.6. Final report

Финальная сводка:
- какая модель лучше;
- где baseline сильнее;
- даёт ли combined setup прирост;
- насколько устойчив результат.

---

## 5. Ограничения среды

Имеем:
- сервер с прерываемыми ресурсами;
- GPU: **RTX 4090**;
- RAM: **32 GB**.

Это означает:
- обучать можно серьёзную модель, но без излишнего раздувания;
- самые опасные места — preprocessing, caching и устойчивость к прерыванию;
- всё сырое чтение CSV должно идти **chunk-wise**;
- основной рабочий формат хранения — **Parquet shards**.

---

## 6. Главная модель: что именно берём

### 6.1. Выбор архитектуры

Для основного решения берём **многоцелевой Transformer encoder для событийных последовательностей**.

Это не «чистый BERT4Rec» в узком смысле и не «чистый SimCLR» в узком смысле.

Это будет:
- **encoder-only Transformer**;
- обучаемый на **raw events**;
- с несколькими задачами обучения:
  1. **masked event modeling**;
  2. **next-event auxiliary head**;
  3. **contrastive prefix consistency loss**.

Идея: получить не просто модель для следующего события, а именно **сильный универсальный encoder**, пригодный для построения эмбеддингов пользователей.

### 6.2. Почему не чистая coarse-token BERT4Rec

Потому что на этом этапе мы сознательно отказываемся от раннего схлопывания событий.

Наша цель:
- научить модель видеть больше структуры;
- сохранить различия между событиями;
- строить эмбеддинг на полном событийном сигнале.

### 6.3. Почему не только SimCLR

Потому что нужен не только downstream-сигнал, но и:
- sequence quality;
- связь с next-event prediction;
- интерпретируемый encoder;
- устойчивость без слишком сложных аугментаций как единственного источника обучения.

### 6.4. Почему не только GRU / LSTM

RNN можно оставить как дополнительный baseline, но основной encoder удобнее делать на Transformer, потому что он:
- лучше подходит под экспорт унифицированных embeddings;
- легче масштабируется на длинные последовательности;
- проще сочетается с masked modeling и contrastive objective.

---

## 7. Представление событий без схлопывания

### 7.1. Базовая единица

Каждое событие представляем как **структурированный объект**, а не как вручную укрупнённый класс.

Минимальный набор полей на событие:
- `event_name`
- `event_json`
- `event_datetime`
- `session_id`
- `user_id / appmetrica_device_id`

### 7.2. Как кодируем событие

Не делаем coarse grouping. Вместо этого строим **структурированную токенизацию**.

Для каждого события формируем:

1. **Event token**
   - базовый токен события;
   - например `event_name`;
   - при необходимости расширенный токен: `event_name + normalized_json_signature`.

2. **Attribute tokens / fields**
   - верхние ключи JSON;
   - важные категориальные поля;
   - категориальные значения из JSON.

3. **Numeric buckets**
   - числовые поля не оставляем сырыми строками;
   - бакетизируем их и кодируем как отдельные признаки.

4. **Time-gap embedding**
   - бакет времени от предыдущего события.

5. **Session boundary flag**
   - отмечаем начало новой сессии.

6. **Positional encoding**
   - стандартная позиционная информация внутри последовательности.

### 7.3. Важное правило

Мы не объединяем события в искусственные укрупнённые группы вроде:
- `reward`
- `economy`
- `meta`

если этого нет в исходных данных как естественной структуры.

Мы сохраняем различимость исходных событий и переносим всю «инженерию» в:
- нормализацию;
- парсинг JSON;
- кодирование признаков;
- архитектуру encoder.

---

## 8. Формат обучения: на чём именно учим

### 8.1. Уровень объекта

Основной объект — **user prefix**.

То есть мы не учим модель на случайных маленьких кусках только ради удобства, а строим реальные пользовательские последовательности событий.

### 8.2. Prefix-based protocol

Для каждого пользователя строим префиксы длины:
- `50`
- `100`
- `150`
- опционально `200`

Эти префиксы будут использоваться:
- и для pretraining;
- и для downstream;
- и для финального сравнения.

### 8.3. Split

Split строго **по пользователям**:
- `train = 70%`
- `valid = 15%`
- `test = 15%`

Один пользователь не может попасть в несколько split.

### 8.4. Без утечки по времени

Downstream labels считаются только по будущему относительно префикса.

Нельзя:
- использовать поздние события для формирования признаков раннего префикса;
- подмешивать в эмбеддер информацию из будущего окна, если потом по нему же строится label.

---

## 9. Основной objective

Основную модель учим по **multitask схеме**.

### 9.1. Loss 1: Masked Event Modeling

Аналог BERT4Rec:
- маскируем часть событий в префиксе;
- модель восстанавливает скрытые события.

Зачем:
- помогает encoder-у понимать контекст;
- хорошо подходит для событийных последовательностей;
- естественно учит полезные представления.

### 9.2. Loss 2: Next-event auxiliary head

Дополнительная задача:
- по префиксу предсказать следующее событие.

Зачем:
- даёт прямую связь с sequence quality;
- делает модель сравнимой с next-event baselines.

### 9.3. Loss 3: Contrastive prefix consistency

Строим две аугментированные версии одного и того же префикса и сближаем их embeddings.

Зачем:
- повышаем устойчивость embeddings;
- учим модель держать инвариантность к слабым шумам и локальным вариациям.

### 9.4. Итоговый loss

Итоговый loss:

`L = w_mlm * L_mlm + w_next * L_next + w_ctr * L_ctr`

Стартовые веса:
- `w_mlm = 1.0`
- `w_next = 0.5`
- `w_ctr = 0.2`

Потом можно тюнить.

---

## 10. Рекомендуемая конфигурация модели

Базовый боевой конфиг для 4090:

- `d_model = 256`
- `n_heads = 8`
- `n_layers = 6`
- `ffn_dim = 1024`
- `dropout = 0.1`
- `max_seq_len = 150` или `200`
- `embedding dropout = 0.1`
- `layer norm = pre-norm`

Если по памяти всё хорошо:
- попробовать `d_model = 384`
- `n_layers = 8`

Но первый полный run лучше делать на стабильной средней конфигурации, а не на максимуме.

---

## 11. Как именно получаем эмбеддинг

Для каждого префикса сохраняем минимум два варианта эмбеддинга:

1. **CLS pooling**
2. **Mean pooling over valid tokens**

Потом сравниваем downstream отдельно:
- `embedding_cls`
- `embedding_mean`

Если один из вариантов явно лучше, его и фиксируем как основной.

---

## 12. Baseline’ы

Нужны baseline’ы двух типов.

### 12.1. Sequence baselines

Это baseline’ы для next-event задачи:
- `MostPopular`
- `Bigram / Markov1`
- `GRU` (опционально)
- `LSTM` (опционально)

Минимум обязательны:
- MostPopular
- Bigram / Markov1

Потому что это самые понятные и сильные последовательностные baseline’ы для сравнения с encoder-моделью.

### 12.2. Downstream baseline

Нужен **сильный feature baseline**, а не декоративный.

Строим user-prefix features:
- count vector по исходным событиям;
- normalized counts;
- top bigram counts;
- число событий;
- число уникальных событий;
- число сессий;
- средняя длина сессии;
- стандартное отклонение длины сессии;
- статистики по time gap;
- энтропия событий;
- повторяемость;
- recency features;
- last-k event indicators.

Дальше два варианта:

#### Вариант A
Прямо кормим эти признаки в:
- Logistic Regression
- LightGBM / XGBoost

#### Вариант B
Строим **baseline embedding**:
- берём feature matrix;
- сжимаем её через `TruncatedSVD(128)` или `256`;
- получаем baseline embedding.

Это нужно, чтобы сравнивать не только «модель vs classifier», но и **embedding vs embedding**.

### 12.3. Combined setup

Обязательно делаем ещё и combined setup:
- `[main_embedding ; baseline_features]`

Потому что в вашей исследовательской логике это важный сценарий: baseline-сигнал и learned embedding могут усиливать друг друга.

---

## 13. Метрики

### 13.1. Sequence quality

Для next-event prediction считаем:
- `MRR@10`
- `HitRate@10`
- опционально `NDCG@10`

### 13.2. Downstream

Для `retention_7d` и `retention_14d` считаем:
- `ROC-AUC`
- `PR-AUC`
- `F1`
- `LogLoss`

Главная таблица должна содержать минимум:
- `prefix_len`
- `target`
- `model_name`
- `roc_auc`
- `pr_auc`

---

## 14. Labels

### 14.1. Обязательное правило

Определение label фиксируется **один раз** в отдельном notebook / конфиге и больше не меняется между моделями.

### 14.2. Что минимум нужно

- `retention_7d`
- `retention_14d`

### 14.3. Как считать

Формально надо зафиксировать одно точное определение.

Например:
- `retention_7d = 1`, если после конца наблюдаемого префикса пользователь вернулся в окно, соответствующее логике 7-дневного retention;
- `retention_14d = 1` аналогично.

Главное здесь не конкретная формула в README, а то, что:
- она должна быть **единой**;
- она должна быть **зафиксирована в коде**;
- она должна быть **одинаковой для всех веток**.

---

## 15. Устойчивость к прерываниям: обязательные правила

### 15.1. Каждый крупный этап пишет готовый артефакт

Нельзя строить проект так, чтобы весь прогресс зависел от непрерывной жизни одной сессии.

После каждого этапа должен появляться физический файл.

### 15.2. Checkpoint policy

Во время обучения обязательно сохраняем:
- `checkpoint_last.pt`
- `checkpoint_best.pt`
- периодические `checkpoint_step_XXXXX.pt`

### 15.3. Что хранить в checkpoint

Каждый checkpoint должен содержать:
- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `scaler_state_dict` если mixed precision
- `epoch`
- `global_step`
- `best_metric`
- `config`
- `split_version`
- `vocab_version`
- `rng_state_python`
- `rng_state_numpy`
- `rng_state_torch_cpu`
- `rng_state_torch_cuda`

### 15.4. Atomic save

Писать checkpoint надо безопасно:
- сначала во временный файл;
- потом атомарный rename.

### 15.5. Resume

При запуске обучения ноутбук обязан:
1. искать `run_state.json`;
2. искать `checkpoint_last.pt`;
3. если они есть — восстанавливаться;
4. если нет — создавать новый run.

### 15.6. Частота сохранения

Стартовая рекомендация:
- каждые `1000` шагов;
- в конце каждой эпохи;
- при улучшении best validation metric.

### 15.7. Отдельный state file

Рядом с checkpoint хранить `run_state.json`:
- `run_id`
- `epoch`
- `global_step`
- `last_checkpoint`
- `best_checkpoint`
- `config_hash`

---

## 16. W&B: нулевая задача

Перед любым тяжёлым обучением надо сделать **нулевую задачу**.

### 16.1. Цель нулевой задачи

Проверить:
- что W&B вообще подключается;
- что run создаётся;
- что метрики логируются;
- что offline/online режим работает;
- что resume работает;
- что чекпоинт сохраняется и подхватывается.

### 16.2. Что должно быть в zero notebook

Notebook `00_wandb_smoke_test.ipynb` должен:
- создать run;
- прологировать 20–50 шагов toy-тренировки;
- записать toy checkpoint;
- уметь перезапуститься и продолжить run по тому же `run_id`.

### 16.3. Что логировать в smoke test

- `train/loss`
- `global_step`
- `epoch`
- `resume_flag`

### 16.4. Критерий успеха

Нулевая задача считается пройденной, если:
- run появляется в интерфейсе или сохраняется локально;
- при рестарте идёт продолжение того же run;
- toy checkpoint успешно загружается.

---

## 17. Структура проекта

```text
project/
├─ notebooks/
│  ├─ 00_wandb_smoke_test.ipynb
│  ├─ 01_ingest_raw_csv_to_parquet.ipynb
│  ├─ 02_parse_and_normalize_events.ipynb
│  ├─ 03_build_vocab.ipynb
│  ├─ 04_build_user_sequences_and_labels.ipynb
│  ├─ 05_train_main_embedder.ipynb
│  ├─ 06_export_main_embeddings.ipynb
│  ├─ 07_build_baseline_features.ipynb
│  ├─ 08_eval_sequence_quality.ipynb
│  ├─ 09_eval_downstream.ipynb
│  └─ 10_final_tables_and_plots.ipynb
│
├─ src/
│  ├─ config.py
│  ├─ io_utils.py
│  ├─ json_parser.py
│  ├─ tokenization.py
│  ├─ datasets.py
│  ├─ model_event_encoder.py
│  ├─ model_rnn_baseline.py
│  ├─ model_markov.py
│  ├─ losses.py
│  ├─ metrics.py
│  ├─ checkpoints.py
│  ├─ wandb_utils.py
│  └─ training.py
│
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  ├─ splits/
│  └─ exports/
│
├─ artifacts/
│  ├─ vocab/
│  ├─ manifests/
│  ├─ checkpoints/
│  ├─ reports/
│  └─ wandb_local/
│
├─ configs/
│  ├─ data.yaml
│  ├─ model.yaml
│  ├─ train.yaml
│  └─ eval.yaml
│
└─ README.md
```

---

## 18. Подробный план ноутбуков

## 18.1. `00_wandb_smoke_test.ipynb`

Цель:
- проверить W&B;
- проверить offline/online;
- проверить run resume;
- проверить сохранение чекпоинта.

На выходе:
- `artifacts/wandb_local/...`
- `artifacts/checkpoints/smoke_checkpoint.pt`
- `artifacts/manifests/smoke_state.json`

---

## 18.2. `01_ingest_raw_csv_to_parquet.ipynb`

Цель:
- читать сырой CSV кусками;
- привести типы;
- сохранить Parquet shards.

Что делает:
1. читает CSV chunk-wise;
2. выкидывает совсем битые строки;
3. приводит даты и id;
4. пишет parquet-шарды.

На выходе:
- `data/interim/events_shard_000.parquet`
- ...
- `artifacts/manifests/ingest_manifest.json`

---

## 18.3. `02_parse_and_normalize_events.ipynb`

Цель:
- разобрать `event_json`;
- сделать канонизацию;
- подготовить структурированное представление событий.

Что делает:
- парсит JSON;
- выделяет категориальные поля;
- бакетизирует числовые поля;
- формирует канонический event record.

Важно:
- не схлопывает события в coarse-классы;
- только нормализует и структурирует.

На выходе:
- `data/interim/events_normalized_*.parquet`

---

## 18.4. `03_build_vocab.ipynb`

Цель:
- построить словари токенов.

Что строим:
- `event_token_vocab.json`
- `json_key_vocab.json`
- `json_value_vocab.json`
- `time_gap_vocab.json`

Дополнительно:
- частоты токенов;
- coverage report;
- OOV strategy.

На выходе:
- `artifacts/vocab/...`

---

## 18.5. `04_build_user_sequences_and_labels.ipynb`

Цель:
- собрать пользовательские последовательности;
- сделать split;
- посчитать labels.

Что делает:
- строит полные последовательности по пользователям;
- режет на prefix lengths `50/100/150`;
- считает `retention_7d`, `retention_14d`;
- сохраняет train/valid/test tables.

На выходе:
- `data/splits/users_train.parquet`
- `data/splits/users_valid.parquet`
- `data/splits/users_test.parquet`
- `data/processed/train_prefixes.parquet`
- `data/processed/valid_prefixes.parquet`
- `data/processed/test_prefixes.parquet`

---

## 18.6. `05_train_main_embedder.ipynb`

Цель:
- обучить основную encoder-модель.

Что делает:
- грузит split и vocab;
- строит dataset/dataloader;
- обучает multitask encoder;
- пишет чекпоинты;
- логирует всё в W&B;
- поддерживает resume.

На выходе:
- `artifacts/checkpoints/checkpoint_last.pt`
- `artifacts/checkpoints/checkpoint_best.pt`
- `artifacts/manifests/run_state.json`

---

## 18.7. `06_export_main_embeddings.ipynb`

Цель:
- выгрузить user embeddings из лучшего чекпоинта.

Что делает:
- загружает `checkpoint_best.pt`;
- прогоняет все префиксы;
- сохраняет embeddings.

На выходе:
- `data/exports/main_embeddings_cls.parquet`
- `data/exports/main_embeddings_mean.parquet`

---

## 18.8. `07_build_baseline_features.ipynb`

Цель:
- построить сильный baseline.

Что делает:
- строит count/bigram/time/session features;
- обучает при необходимости `TruncatedSVD`;
- сохраняет baseline features и baseline embeddings.

На выходе:
- `data/exports/baseline_features.parquet`
- `data/exports/baseline_embeddings_128d.parquet`

---

## 18.9. `08_eval_sequence_quality.ipynb`

Цель:
- оценить качество на next-event задаче.

Что сравниваем:
- MostPopular
- Bigram / Markov1
- GRU / LSTM (если будут)
- Main encoder model

На выходе:
- `artifacts/reports/sequence_metrics.csv`

---

## 18.10. `09_eval_downstream.ipynb`

Цель:
- оценить downstream.

Что сравниваем:
- baseline features + linear model
- baseline features + tree model
- main embeddings + linear model
- main embeddings + MLP
- combined setup

На выходе:
- `artifacts/reports/downstream_metrics.csv`

---

## 18.11. `10_final_tables_and_plots.ipynb`

Цель:
- собрать финальные таблицы и графики для курсовой / отчёта / презы.

На выходе:
- `artifacts/reports/final_results.xlsx`
- `artifacts/reports/final_plots/*.png`

---

## 19. Формат данных между этапами

### 19.1. После ingestion

Одна строка = одно сырое событие.

### 19.2. После normalization

Одна строка = одно нормализованное событие с разобранными атрибутами.

### 19.3. После sequence build

Одна строка = один user-prefix.

Поля:
- `user_id`
- `prefix_len`
- `event_token_ids`
- `attribute_ids`
- `time_gap_ids`
- `session_flags`
- `label_retention_7d`
- `label_retention_14d`
- `split`

### 19.4. После export embeddings

Одна строка = один embedding одного user-prefix.

---

## 20. Детали обучения

### 20.1. Batch strategy

Если по памяти тяжело:
- используем `batch_size` меньше;
- добавляем `gradient_accumulation`.

### 20.2. Mixed precision

Включаем mixed precision, если сборка и среда стабильны.

### 20.3. Оптимизатор

Стартово:
- `AdamW`
- `lr = 2e-4`
- `weight_decay = 1e-2`

### 20.4. Scheduler

- linear warmup
- cosine decay или linear decay

### 20.5. Early stopping

Останавливаемся по `valid ROC-AUC` на downstream probe или по заранее выбранной основной метрике.

---

## 21. Что логировать в W&B

### 21.1. Train

- `train/loss_total`
- `train/loss_mlm`
- `train/loss_next`
- `train/loss_ctr`
- `train/lr`
- `train/grad_norm`
- `train/tokens_seen`
- `train/global_step`

### 21.2. Valid

- `valid/loss_total`
- `valid/mrr_at_10`
- `valid/hit_at_10`
- `valid/retention_7d_auc`
- `valid/retention_14d_auc`

### 21.3. Artifacts

- vocab
- split
- best checkpoint
- embeddings export
- final tables

---

## 22. Что считать успешным первым полным запуском

Первый full run считается успешным, если:

1. ingestion завершился;
2. словарь построен;
3. split сохранён;
4. labels сохранены;
5. основная модель обучилась хотя бы до первой стабильной валидации;
6. есть `checkpoint_best.pt`;
7. можно выгрузить embeddings;
8. baseline features построены;
9. собрана таблица sequence metrics;
10. собрана таблица downstream metrics.

---

## 23. Что будет считаться хорошим результатом

### Минимально хороший результат

- основная модель даёт осмысленные embeddings;
- embeddings работают в downstream не хуже простого baseline.

### Хороший результат

- основная модель бьёт хотя бы один сильный baseline на части метрик;
- combined setup даёт прирост.

### Очень хороший результат

- main embeddings выигрывают на downstream;
- sequence quality при этом остаётся конкурентной;
- результаты устойчиво воспроизводятся после resume.

---

## 24. Что принципиально нельзя делать

Нельзя:
- снова переходить к маленькой игрушечной подвыборке как к основному режиму;
- обучать без сохранения чекпоинтов;
- полагаться на состояние RAM;
- менять split между моделями;
- менять definition labels между ветками;
- сравнивать модели по разным наборам пользователей;
- схлопывать события «для удобства» на этом этапе.

---

## 25. Итоговое решение по проекту

На этом этапе фиксируем следующую стратегию:

### Основная модель

**Многоцелевой Transformer encoder на raw event sequences**
без coarse event collapsing.

### Baseline’ы

- `MostPopular`
- `Bigram / Markov1`
- strong feature baseline
- optional GRU / LSTM

### Основной протокол

- split по пользователям
- prefix lengths `50 / 100 / 150`
- downstream targets `retention_7d / retention_14d`
- sequence metrics `MRR@10 / HitRate@10`
- downstream metrics `ROC-AUC / PR-AUC`

### Инженерная стратегия

- parquet shards
- manifest files
- frequent checkpoints
- W&B logging
- full resume support
- notebook-by-stage pipeline

---

## 26. Самый короткий operational summary

Если совсем кратко:

1. делаем W&B smoke test;
2. превращаем сырой CSV в parquet shards;
3. парсим и нормализуем raw events без coarse-collapsing;
4. строим vocab и user-prefix dataset;
5. обучаем multitask Transformer encoder на всём train split;
6. сохраняем чекпоинты и умеем resume;
7. экспортируем embeddings;
8. строим strong baseline features / baseline embeddings;
9. сравниваем всё по sequence и downstream метрикам;
10. получаем финальную таблицу для курсовой.

