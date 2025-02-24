# ✅ CUDA & Ubuntu 20.04 ベースの Python イメージ
FROM nvidia/cuda:12.8.0-base-ubuntu20.04

# 作業ディレクトリを設定
WORKDIR /app

# Python & 必要なパッケージをインストール
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Poetry のインストール
RUN pip3 install poetry

# Poetry の仮想環境を無効化（グローバル環境にインストールするため）
RUN poetry config virtualenvs.create false

# ✅ `pyproject.toml` & `poetry.lock` をコピー
COPY ./pyproject.toml ./poetry.lock ./
RUN ls -la /app

# ✅ 依存関係をインストール
RUN poetry install --no-root

# ✅ `backend/` のコードをコピー
COPY backend/ /app/backend/

# ✅ `data/` を必要に応じてコピー
COPY data/ /app/data/
COPY dataset/ /app/dataset/

# ✅ 環境変数で GPU を有効化
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# FastAPI を起動
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
