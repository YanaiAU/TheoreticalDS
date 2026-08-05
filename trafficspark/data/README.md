# TrafficSpark data

Combined file `accidents_2020_2024.csv` is Israel CBS **Public Use File** road accidents with casualties, 2020–2024 (~49.9k rows).

- Official package: https://data.gov.il/dataset/02789da8-7a3e-4bfc-b771-1732b1cf403c  
- Overview: https://govil.ai/datasets/02789da8-7a3e-4bfc-b771-1732b1cf403c/

Target: `HUMRAT_TEUNA` — 1=fatal, 2=severe, 3=light.

Reproduce download:

```bash
python src/download_data.py
```
