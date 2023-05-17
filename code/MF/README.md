### Directory Structure

```bash
MF
|-- README.md
|-- MF_util
|   |-- __init__.py
|   |-- args.py         # Argument
|   |-- datasets.py     # Datasets for CatBoost
|   |-- models.py       # NMF, SVD Model Class
|   `-- utils.py        # Common Utils
|-- main.py             # dataset load -> split -> model load -> predict&evaluate
|-- poetry.lock         # poetry
`-- pyproject.toml      # poetry
```

### RUN

```python
cd code/MF & poetry install
python main.py
# "--seed", default=42, type=int, help="랜덤 시드를 지정합니다."
# "--output_dir", default="./outputs/", type=str, help="predict 결과를 저장할 경로를 지정합니다."
# "--data_dir", default="/opt/ml/input/data", type=str, help="데이터 경로를 지정합니다."
# "--model", default="NMF", choices=["SVD", "NMF"], help="SVD, NMF 모델 중 하나를 선택합니다."
# "--k", default=12, type=int, help="잠재행렬의 요소 수(k)를 지정합니다."
```

### OUTPUT BASH
- SVD

    ![image](https://file.notion.so/f/s/ff350ef1-840d-4eeb-a606-337799bdf649/Untitled.png?id=3487f40d-b31e-456f-bd89-5be2d3e16c3b&table=block&spaceId=1969fbfa-c4f5-4e52-8ad0-ef9dbe665f2f&expirationTimestamp=1684162695228&signature=mYaaD-1-WBYjzw379O9agpxmYFSPpJmM09Pt7gdDpFs&downloadName=Untitled.png)

- NMF

    ![image](https://file.notion.so/f/s/9fc6acab-b182-4ff8-acd7-ff1a2a77dd47/Untitled.png?id=34700ae2-4f26-4c85-8e81-d95c1f7445e1&table=block&spaceId=1969fbfa-c4f5-4e52-8ad0-ef9dbe665f2f&expirationTimestamp=1684162788428&signature=mYwL9IAMvd99iC5vq37zjmOZ4naX7_l3AuZHYOr9RIE&downloadName=Untitled.png)
