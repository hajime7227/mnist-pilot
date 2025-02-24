# ✅ Python ベースの Streamlit イメージ
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# Poetry のインストール
RUN pip install poetry

# Poetry の仮想環境を無効化
RUN poetry config virtualenvs.create false

# 依存関係のインストール
COPY ./pyproject.toml ./poetry.lock ./
RUN poetry install --no-dev

# `frontend/` のコードをコピー
COPY frontend/ /app/frontend/

# Streamlit を起動
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
