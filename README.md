# nlp-gateway

**English** | [日本語](#日本語) | [中文](#中文)

A multi-capability NLP microservice gateway built with FastAPI, serving as the unified inference backend for a conversational AI system.

## Architecture

```
app.py                          # FastAPI entry, router registration, startup banner
routers/                        # HTTP layer — request/response models, validation
  ├── judge.py                  # Promise acceptance judgment
  ├── time_parse.py             # Temporal expression parsing
  ├── features_promise.py       # Promise candidate feature extraction
  ├── hard_write.py             # Explicit memory-write command parsing
  ├── embed_qualify.py          # Embedding quality gate
  ├── embed_encode.py           # Text → vector encoding
  ├── emotion.py                # Single-sentence emotion classification
  └── tts.py                    # Text-to-speech synthesis
services/                       # Business logic — models, NLP pipelines, clients
  ├── judge_logic.py            # Regex-based acceptance/rejection detection
  ├── duckling_client.py        # Facebook Duckling temporal parser client
  ├── promise_features.py       # Multi-signal promise detection (time + action + intent)
  ├── hard_write_logic.py       # Profile field extraction via HanLP + BERT classifier
  ├── profile_parse.py          # User profile parsing (HanLP tokenization + V-O extraction)
  ├── embedding_service.py      # Sentence-Transformers lazy-loaded encoder
  ├── emotion_service.py        # MacBERT 6-class emotion classifier
  ├── tts_service.py            # OpenAI TTS wrapper with stage-direction removal
  ├── lang_guess.py             # Language detection heuristic
  ├── nlp.py                    # spaCy NLP utilities
  └── message_filter/           # Strategy pattern — extensible per-language filtering
        ├── base.py             # Abstract FilterStrategy + FilterResult
        └── chinese.py          # Chinese filter: HanLP SRL/POS scoring + fast-path regex
models/                         # Pre-trained model artifacts
  ├── emotion_model/            # Fine-tuned MacBERT for 6-class emotion
  └── profile_model/            # Fine-tuned MacBERT for profile field classification
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI + Uvicorn |
| Chinese NLP | HanLP (tokenization, POS, NER, SRL) |
| Emotion Classification | MacBERT fine-tuned (6 classes: neutral / happy / sad / angry / proud / shy) |
| Profile Classification | MacBERT fine-tuned (multi-label profile field detection) |
| Embeddings | Sentence-Transformers |
| Temporal Parsing | Facebook Duckling |
| TTS | OpenAI gpt-4o-mini-tts |
| General NLP | spaCy |

## Design Highlights

- **Router-Service Separation** — Clean HTTP layer (routers) decoupled from business logic (services). Each router only handles request validation and response formatting.

- **Lazy Loading** — All heavy models (HanLP, MacBERT, Sentence-Transformers) are loaded on first use, not at startup. This keeps cold start fast and memory efficient when not all endpoints are needed.

- **Strategy Pattern** — Message filtering uses an abstract `FilterStrategy` base class with language-specific implementations (`ChineseFilterStrategy`). New languages can be added by implementing the interface.

- **Trace ID Propagation** — Every request carries a `x-trace-id` header, logged consistently across all services for end-to-end debugging.

- **Graceful Fallback** — Missing model weights (`.safetensors`) don't crash the server. Affected endpoints return 503 while all other endpoints remain operational.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/judge/accept` | POST | Promise acceptance judgment |
| `/features/time/parse` | POST | Temporal expression parsing (via Duckling) |
| `/features/promise/candidate` | POST | Promise candidate feature extraction |
| `/hard_write/judge` | POST | Explicit write-command parsing |
| `/embed/qualify` | POST | Embedding quality gate (SRL + POS scoring) |
| `/embed/encode` | POST | Text → vector encoding |
| `/emotion/analyze` | POST | Single-sentence emotion classification |
| `/tts/speak` | POST | Text-to-speech (returns MP3 audio stream) |

## Pre-trained Models

Model weight files (`.safetensors`) are excluded from git. After cloning, copy them manually:

```bash
# emotion_model — 6-class emotion classifier (MacBERT fine-tuned)
cp /path/to/emotion_model/model.safetensors models/emotion_model/

# profile_model — profile field classifier (MacBERT fine-tuned)
cp /path/to/profile_model/model.safetensors models/profile_model/
```

Config and tokenizer files are tracked in git.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env  # then fill in OPENAI_API_KEY, etc.

# 4. Run
uvicorn app:app --host 127.0.0.1 --port 8123 --log-level info
```

## Deployment

This service runs on a **Raspberry Pi 4B (ARM64, CPU-only)** as part of a larger conversational AI system.

**Runtime dependencies:**

| Service | Purpose | Default Address |
|---------|---------|-----------------|
| nlp-gateway (this) | NLP inference | `http://127.0.0.1:8123` |
| Facebook Duckling | Temporal expression parsing | `http://127.0.0.1:8001` |

Duckling is deployed as a separate process on the same host. It provides structured time/date extraction (e.g. "下周三下午三点" → ISO 8601) used by the `/features/time/parse` and `/features/promise/candidate` endpoints.

**Resource considerations:**

- All ML models run on CPU — PyTorch is installed with `--index-url https://download.pytorch.org/whl/cpu` to minimize binary size
- Embedding model (`BAAI/bge-small-zh-v1.5`, ~95MB) chosen for low memory footprint
- Lazy loading ensures only requested models are loaded into memory (~2-3GB peak when all models are active)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For TTS | OpenAI API key for text-to-speech |
| `DUCKLING_URL` | Optional | Duckling server URL (default: `http://127.0.0.1:8001/parse`) |
| `LOG_LEVEL` | Optional | Logging level (default: `INFO`) |

---

# 日本語

**[English](#nlp-gateway) | 日本語 | [中文](#中文)**

会話型AIシステムの統合推論バックエンドとして機能する、FastAPIベースのマルチ機能NLPマイクロサービスゲートウェイです。

## アーキテクチャ

```
app.py                          # FastAPIエントリ、ルーター登録、起動バナー
routers/                        # HTTPレイヤー — リクエスト/レスポンスモデル、バリデーション
  ├── judge.py                  # 約束承諾判定
  ├── time_parse.py             # 時間表現パーシング
  ├── features_promise.py       # 約束候補の特徴抽出
  ├── hard_write.py             # 明示的メモリ書込コマンド解析
  ├── embed_qualify.py          # 埋め込み品質ゲート
  ├── embed_encode.py           # テキスト → ベクトル変換
  ├── emotion.py                # 単文感情分類
  └── tts.py                    # テキスト音声合成
services/                       # ビジネスロジック — モデル、NLPパイプライン、クライアント
  ├── judge_logic.py            # 正規表現ベースの承諾/拒否検出
  ├── duckling_client.py        # Facebook Duckling 時間パーサークライアント
  ├── promise_features.py       # マルチシグナル約束検出（時間+行動+意図）
  ├── hard_write_logic.py       # HanLP + BERT分類器によるプロフィールフィールド抽出
  ├── profile_parse.py          # ユーザープロフィール解析（HanLPトークン化 + 動目的語抽出）
  ├── embedding_service.py      # Sentence-Transformers 遅延読込エンコーダー
  ├── emotion_service.py        # MacBERT 6クラス感情分類器
  ├── tts_service.py            # OpenAI TTS ラッパー（ト書き除去機能付き）
  ├── lang_guess.py             # 言語検出ヒューリスティック
  ├── nlp.py                    # spaCy NLPユーティリティ
  └── message_filter/           # Strategyパターン — 言語別拡張可能フィルタリング
        ├── base.py             # 抽象FilterStrategy + FilterResult
        └── chinese.py          # 中国語フィルター: HanLP SRL/POSスコアリング + 高速パス正規表現
models/                         # 事前学習済みモデル成果物
  ├── emotion_model/            # ファインチューニング済みMacBERT（6クラス感情）
  └── profile_model/            # ファインチューニング済みMacBERT（マルチラベルプロフィール分類）
```

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| フレームワーク | FastAPI + Uvicorn |
| 中国語NLP | HanLP（トークン化、品詞タグ、固有表現認識、意味役割付与） |
| 感情分類 | MacBERT ファインチューニング（6クラス: 平常 / 嬉しい / 悲しい / 怒り / 得意 / 恥ずかしい） |
| プロフィール分類 | MacBERT ファインチューニング（マルチラベルプロフィールフィールド検出） |
| 埋め込み | Sentence-Transformers |
| 時間パーシング | Facebook Duckling |
| TTS | OpenAI gpt-4o-mini-tts |
| 汎用NLP | spaCy |

## 設計のポイント

- **Router-Service分離** — HTTPレイヤー（routers）とビジネスロジック（services）を明確に分離。各ルーターはリクエスト検証とレスポンス整形のみを担当。

- **遅延読込（Lazy Loading）** — 全重量モデル（HanLP、MacBERT、Sentence-Transformers）は初回使用時にロード。コールドスタートの高速化と、不要なエンドポイントのメモリ節約を実現。

- **Strategyパターン** — メッセージフィルタリングは抽象基底クラス `FilterStrategy` と言語別実装（`ChineseFilterStrategy`）で構成。インターフェースの実装のみで新言語を追加可能。

- **Trace ID伝播** — 全リクエストに `x-trace-id` ヘッダーを付与し、全サービスで一貫してログ出力。エンドツーエンドのデバッグに対応。

- **グレースフルフォールバック** — モデルの重みファイル（`.safetensors`）が欠落してもサーバーはクラッシュしない。該当エンドポイントのみ503を返し、他は正常稼働を継続。

## APIエンドポイント

| エンドポイント | メソッド | 説明 |
|--------------|---------|------|
| `/health` | GET | ヘルスチェック |
| `/judge/accept` | POST | 約束承諾判定 |
| `/features/time/parse` | POST | 時間表現パーシング（Duckling経由） |
| `/features/promise/candidate` | POST | 約束候補の特徴抽出 |
| `/hard_write/judge` | POST | 明示的書込コマンド解析 |
| `/embed/qualify` | POST | 埋め込み品質ゲート（SRL + POSスコアリング） |
| `/embed/encode` | POST | テキスト → ベクトル変換 |
| `/emotion/analyze` | POST | 単文感情分類 |
| `/tts/speak` | POST | テキスト音声合成（MP3ストリーム返却） |

## 事前学習済みモデル

モデルの重みファイル（`.safetensors`）はgitから除外しています。クローン後、手動でコピーしてください：

```bash
# emotion_model — 6クラス感情分類器（MacBERTファインチューニング済み）
cp /path/to/emotion_model/model.safetensors models/emotion_model/

# profile_model — プロフィールフィールド分類器（MacBERTファインチューニング済み）
cp /path/to/profile_model/model.safetensors models/profile_model/
```

設定ファイルとトークナイザーファイルはgitで管理しています。

## クイックスタート

```bash
# 1. 仮想環境を作成
python -m venv .venv
source .venv/bin/activate

# 2. 依存関係をインストール
pip install -r requirements.txt

# 3. 環境変数を設定
cp .env.example .env  # OPENAI_API_KEYなどを記入

# 4. 起動
uvicorn app:app --host 127.0.0.1 --port 8123 --log-level info
```

## デプロイ

本サービスは会話型AIシステムの一部として、**Raspberry Pi 4B（ARM64、CPUのみ）** 上で稼働しています。

**ランタイム依存サービス：**

| サービス | 用途 | デフォルトアドレス |
|---------|------|------------------|
| nlp-gateway（本サービス） | NLP推論 | `http://127.0.0.1:8123` |
| Facebook Duckling | 時間表現パーシング | `http://127.0.0.1:8001` |

Ducklingは同一ホスト上に別プロセスとしてデプロイ。構造化された日時抽出（例："下周三下午三点" → ISO 8601）を提供し、`/features/time/parse` と `/features/promise/candidate` エンドポイントで使用されています。

**リソース面の考慮事項：**

- 全MLモデルはCPU上で推論 — PyTorchは `--index-url https://download.pytorch.org/whl/cpu` でインストールしバイナリサイズを最小化
- 埋め込みモデル（`BAAI/bge-small-zh-v1.5`、約95MB）は低メモリフットプリントを考慮して選定
- 遅延読込により、リクエストされたモデルのみメモリにロード（全モデル稼働時のピークは約2〜3GB）

## 環境変数

| 変数名 | 必須 | 説明 |
|-------|------|------|
| `OPENAI_API_KEY` | TTS使用時 | OpenAI APIキー（テキスト音声合成用） |
| `DUCKLING_URL` | 任意 | DucklingサーバーURL（デフォルト: `http://127.0.0.1:8001/parse`） |
| `LOG_LEVEL` | 任意 | ログレベル（デフォルト: `INFO`） |

---

# 中文

**[English](#nlp-gateway) | [日本語](#日本語) | 中文**

基于 FastAPI 构建的多功能 NLP 微服务网关，作为会话式 AI 系统的统一推理后端。

## 架构

```
app.py                          # FastAPI 入口、路由注册、启动日志
routers/                        # HTTP 层 — 请求/响应模型、参数校验
  ├── judge.py                  # 承诺接受判定
  ├── time_parse.py             # 时间表达式解析
  ├── features_promise.py       # 承诺候选特征提取
  ├── hard_write.py             # 显式记忆写入命令解析
  ├── embed_qualify.py          # 嵌入质量门控
  ├── embed_encode.py           # 文本 → 向量编码
  ├── emotion.py                # 单句情感分类
  └── tts.py                    # 文本转语音
services/                       # 业务逻辑 — 模型、NLP 管线、外部客户端
  ├── judge_logic.py            # 基于正则的接受/拒绝检测
  ├── duckling_client.py        # Facebook Duckling 时间解析客户端
  ├── promise_features.py       # 多信号承诺检测（时间+动作+意图）
  ├── hard_write_logic.py       # HanLP + BERT 分类器提取个人资料字段
  ├── profile_parse.py          # 用户资料解析（HanLP 分词 + 动宾提取）
  ├── embedding_service.py      # Sentence-Transformers 懒加载编码器
  ├── emotion_service.py        # MacBERT 6 类情感分类器
  ├── tts_service.py            # OpenAI TTS 封装（含舞台指令移除）
  ├── lang_guess.py             # 语言检测启发式算法
  ├── nlp.py                    # spaCy NLP 工具
  └── message_filter/           # 策略模式 — 按语言可扩展的消息过滤
        ├── base.py             # 抽象 FilterStrategy + FilterResult
        └── chinese.py          # 中文过滤器：HanLP SRL/POS 打分 + 快速路径正则
models/                         # 预训练模型文件
  ├── emotion_model/            # 微调 MacBERT（6 类情感）
  └── profile_model/            # 微调 MacBERT（多标签资料分类）
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 框架 | FastAPI + Uvicorn |
| 中文 NLP | HanLP（分词、词性标注、命名实体识别、语义角色标注） |
| 情感分类 | MacBERT 微调（6 类：平常 / 开心 / 伤心 / 生气 / 得意 / 害羞） |
| 资料分类 | MacBERT 微调（多标签个人资料字段检测） |
| 嵌入 | Sentence-Transformers |
| 时间解析 | Facebook Duckling |
| TTS | OpenAI gpt-4o-mini-tts |
| 通用 NLP | spaCy |

## 设计亮点

- **Router-Service 分离** — HTTP 层（routers）与业务逻辑（services）解耦。每个路由仅负责请求校验和响应格式化。

- **懒加载** — 所有重量级模型（HanLP、MacBERT、Sentence-Transformers）在首次使用时才加载，而非启动时。冷启动快、不使用的端点不占内存。

- **策略模式** — 消息过滤采用抽象基类 `FilterStrategy` 搭配语言特定实现（`ChineseFilterStrategy`）。只需实现接口即可添加新语言支持。

- **Trace ID 透传** — 每个请求携带 `x-trace-id` 头，在所有服务层一致地记录日志，支持端到端调试。

- **优雅降级** — 模型权重文件（`.safetensors`）缺失不会导致服务崩溃。受影响的端点返回 503，其余端点正常运行。

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/judge/accept` | POST | 承诺接受判定 |
| `/features/time/parse` | POST | 时间表达式解析（通过 Duckling） |
| `/features/promise/candidate` | POST | 承诺候选特征提取 |
| `/hard_write/judge` | POST | 显式写入命令解析 |
| `/embed/qualify` | POST | 嵌入质量门控（SRL + POS 打分） |
| `/embed/encode` | POST | 文本 → 向量编码 |
| `/emotion/analyze` | POST | 单句情感分类 |
| `/tts/speak` | POST | 文本转语音（返回 MP3 音频流） |

## 预训练模型

模型权重文件（`.safetensors`）已从 git 中排除。克隆后请手动复制：

```bash
# emotion_model — 6 类情感分类器（MacBERT 微调）
cp /path/to/emotion_model/model.safetensors models/emotion_model/

# profile_model — 资料字段分类器（MacBERT 微调）
cp /path/to/profile_model/model.safetensors models/profile_model/
```

配置文件和分词器文件已纳入 git 管理。

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 设置环境变量
cp .env.example .env  # 填入 OPENAI_API_KEY 等

# 4. 启动
uvicorn app:app --host 127.0.0.1 --port 8123 --log-level info
```

## 部署

本服务作为会话式 AI 系统的一部分，运行在 **Raspberry Pi 4B（ARM64，仅 CPU）** 上。

**运行时依赖服务：**

| 服务 | 用途 | 默认地址 |
|------|------|---------|
| nlp-gateway（本服务） | NLP 推理 | `http://127.0.0.1:8123` |
| Facebook Duckling | 时间表达式解析 | `http://127.0.0.1:8001` |

Duckling 作为独立进程部署在同一主机上，提供结构化的日期时间提取（例如："下周三下午三点" → ISO 8601），供 `/features/time/parse` 和 `/features/promise/candidate` 端点使用。

**资源优化考量：**

- 所有 ML 模型在 CPU 上推理 — PyTorch 通过 `--index-url https://download.pytorch.org/whl/cpu` 安装以最小化二进制体积
- 嵌入模型（`BAAI/bge-small-zh-v1.5`，约 95MB）选型考虑了低内存占用
- 懒加载确保仅将请求到的模型加载进内存（全部模型活跃时峰值约 2~3GB）

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | TTS 功能需要 | OpenAI API 密钥（文本转语音） |
| `DUCKLING_URL` | 可选 | Duckling 服务器地址（默认：`http://127.0.0.1:8001/parse`） |
| `LOG_LEVEL` | 可选 | 日志级别（默认：`INFO`） |
