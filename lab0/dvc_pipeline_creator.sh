dvc stage add -n preprocess \
    -d data/raw/train.csv \
    -d src/preprocessing.py \
    -o data/interim/train.csv \
    -o data/interim/val.csv \
    -o data/processed/train.csv \
    -o data/processed/val.csv \
    -o models/random_forest/columns_transformer.pkl \
    -p paths \
    -p dataset \
    -p names \
    -p hyperparameters \
    -p tuning \
    -p pipeline_config \
    --force \
    uv run src/preprocessing.py

