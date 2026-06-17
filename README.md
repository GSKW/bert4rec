# BERT4Rec User Embeddings

Проект строит пользовательские эмбеддинги по последовательностям игровых событий. Финальная ветка проекта описывает BERT/Transformer-подход в режиме **full-history last-512**: для каждого пользователя берутся последние 512 событий полной доступной истории, модель обучается на next-event/MLM/contrastive сигналах, а качество эмбеддингов проверяется на задачах удержания.

## Что реализовано

- подготовка событий, словаря токенов и пользовательских последовательностей;
- Transformer/BERT encoder для последовательностей событий;
- экспорт пользовательских эмбеддингов в вариантах `cls`, `mean` и `readout`;
- baseline-представления для сравнения;
- downstream-оценка на `retention_7d`, `retention_14d`, `retention_30d`;
- sequence-оценка next-event prediction.

Финальный протокол согласован с отчетом и презентацией:

- общий split пользователей берется из `master_split.csv`;
- окно модели: `full-history last-512`;
- label retention считается по полной наблюдаемой истории пользователя: `last_event_ts - first_event_ts >= N days`;
- основной вариант для отчета: `main_mean_logreg` поверх mean-pooling эмбеддинга.

## Структура

```text
configs/      YAML-конфиги этапов пайплайна
notebooks/    пошаговые ноутбуки для запуска этапов
scripts/      воспроизводимые entrypoint-скрипты
src/          основной Python-код пайплайна
artifacts/    локальные результаты запусков; в git хранятся только .gitkeep/легкие файлы
data/         локальные данные и parquet-экспорты; не входят в git
```

Большие данные, checkpoints, W&B-логи, parquet-экспорты, архивы и промежуточные отчеты считаются локальными артефактами и не должны попадать в репозиторий.

## Установка

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Для полного запуска нужны исходные события `events.csv`, файл split `master_split.csv`, GPU и достаточно места под промежуточные parquet-файлы.

## Основной запуск

```powershell
python scripts\run_full_history_bert.py `
  --run-name bert_full_history_last512 `
  --max-history-len 512 `
  --retention-days 7 14 30 `
  --downstream-targets retention_7d retention_14d retention_30d `
  --batch-size 4 `
  --eval-batch-size 8 `
  --num-workers 2 `
  --wandb-mode online
```

Скрипт собирает full-history датасеты, обучает encoder, экспортирует эмбеддинги, строит baseline и пересчитывает downstream-метрики.

## Итоговые метрики отчета

Ключевой результат для BERT mean pooling:

| Target | Test positive rate | ROC-AUC | PR-AUC | F1 | LogLoss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `retention_7d` | 0.1108 | 0.9296 | 0.6228 | 0.5307 | 0.3550 |
| `retention_14d` | 0.0598 | 0.9256 | 0.4467 | 0.3819 | 0.3582 |
| `retention_30d` | 0.0034 | 0.9665 | 0.1500 | 0.0742 | 0.1966 |

Для `retention_30d` положительный класс очень редкий, поэтому ROC-AUC нужно читать вместе с PR-AUC и positive rate.

## Проверка кода

```powershell
python -m py_compile src\*.py scripts\*.py
```
