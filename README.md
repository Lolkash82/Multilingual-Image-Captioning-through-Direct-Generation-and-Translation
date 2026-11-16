# OpenFlamingo baseline: воспроизведение работы M. Duračka (2026)

Код воспроизводит пять режимов из таблицы 5-3 диссертации на тестовой части Multi30K 2016
(1 000 изображений): EN-only, Direct DE, Direct FR, Translate DE, Translate FR.
Результаты служат базой для сравнения с Qwen2.5-VL-7B и другими способами перевода.

Код написан заново по описанию в диссертации (главы 4 и 5, листинги 1 и 2).
Оригинального репозитория автора здесь нет, поэтому небольшие расхождения с его цифрами возможны.

## Запуск на Perun

```bash
# 1. На login-узле (нужен интернет)
bash scripts/setup_env.sh                 # venv, torch, transformers==4.44.2, OpenFlamingo --no-deps
source venv-flamingo/bin/activate
export HF_HOME=$PWD/.hf_cache
python scripts/prepare_multi30k.py        # подписи (GitHub) + изображения Flickr30K (Hugging Face, ~4.4 GB)
python scripts/download_models.py         # все модели в кэш, чтобы узлы с GPU работали офлайн
python -m pytest tests -q                 # 6 тестов без GPU

# 2. Пробный запуск на 20 картинках (по ~минуте на режим)
LIMIT=20 sbatch scripts/slurm/01_flamingo.sh

# 3. Полный прогон: генерация -> перевод -> оценка
bash scripts/slurm/run_all.sh
```

Перед запуском проверь в `scripts/slurm/*.sh` имя раздела (`--partition=GPU`), запрос GPU и лимиты
под свой аккаунт. Если на узлах с GPU есть интернет, убери `HF_HUB_OFFLINE` из `scripts/slurm/common.sh`.

## Что получится

| Файл | Содержимое |
|---|---|
| `outputs/captions/{en,direct_de,direct_fr}.json` | подписи (очищенные и сырые), время на картинку, референсы |
| `outputs/translations/translate_{de,fr}.json` | переводы MarianMT |
| `outputs/metrics/summary.csv` | BLEU, chrF, METEOR, BERTScore, доля нужного языка, сек/подпись |
| `outputs/metrics/comparison_with_thesis.csv` | разница с таблицей 5-3 диссертации |
| `outputs/metrics/metrics.json` | всё подробно: распределение языков, длины, brevity penalty, подписи sacreBLEU |

Прерванное задание можно просто отправить заново: готовые подписи читаются из `*.partial.jsonl`.

## Что взято из диссертации, а что предположено

Из диссертации: модель OpenFlamingo-3B-vitl-mpt1b, 4 few-shot примера из train с seed 42,
шаблоны `Output:` / `German description:` / `French description:`, beam search (3 луча, 50 новых токенов),
постобработка по листингу 2, MarianMT (батч 64, 4 луча, длина 128), модели BERTScore, langdetect.

Предположено (в диссертации не указано), всё вынесено в `configs/config.yaml`:
- **Способ выбора few-shot примеров:** `random.Random(42).sample(train, 4)`. Если автор выбирал иначе,
  примеры будут другими, и цифры сдвинутся. Список выбранных картинок пишется в `meta.few_shot_images`.
- **Размер батча Flamingo:** 1. Промпт одинаков для всех картинок, поэтому батч можно увеличить
  (`--batch-size 16`) без паддинга на входе; время на подпись тогда не сравнимо с таблицей 5-2.
- **METEOR:** NLTK, среднее по предложениям. Синонимы берутся из английского WordNet, поэтому для DE/FR
  метрика считает в основном точные совпадения.
- **langdetect:** зафиксирован seed 0 (без него результат меняется от запуска к запуску).
- **Точность:** fp32.
