# Data

Real corpus built from Hugging Face [`aadityaubhat/GPT-wiki-intro`](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro):

- **150,000** Wikipedia topics
- For each topic: the real Wikipedia lead intro + a GPT (Curie) generated intro
- Flattened into binary rows: `label=0` human, `label=1` AI

```bash
python src/download_data.py              # default: 75k topics → ~150k texts
python src/download_data.py --max-topics 150000   # full corpus → ~300k texts
```

Output: `wiki_ai_detection.csv` (gitignored — regenerate locally).

| column     | meaning                                      |
|------------|----------------------------------------------|
| topic_id   | original dataset id                          |
| title      | Wikipedia page title                         |
| url        | Wikipedia URL                                |
| text       | intro paragraph                              |
| label      | 0 = human Wikipedia, 1 = GPT-generated       |
| source     | `wikipedia` or `gpt_curie`                   |
| word_count | word count from the upstream dataset         |
