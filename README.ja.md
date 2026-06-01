<div align="center">
  <img
    src="https://github.com/user-attachments/assets/aa171a4c-074c-4082-b3d1-c70f5f7f2aca"
    alt="XMem Logo"
    width="100%"
  />
</div>

<div align="center">
  <h1>XMem</h1>
  <p><strong>決して忘れない AI のためのメモリレイヤー</strong></p>
  <p>すべての AI エージェントと LLM インターフェースに、永続的でクロスプラットフォームなメモリをすぐに提供します。</p>

<img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+"/>
<img src="https://img.shields.io/badge/license-BSD--3--Clause-green" alt="BSD-3 License"/>
<img src="https://img.shields.io/badge/FastAPI-00C7B7?logo=fastapi&logoColor=white" alt="FastAPI"/>
<br/>
<img src="https://img.shields.io/badge/LangGraph-6C47FF?logo=langchain&logoColor=white" alt="LangGraph"/>
<img src="https://img.shields.io/badge/Rust-Weaver-b7410e?logo=rust&logoColor=white" alt="Rust Weaver"/>
<img src="https://img.shields.io/badge/Multi--LLM-Gemini%20%7C%20Claude%20%7C%20GPT%20%7C%20Bedrock%20%7C%20Ollama-orange" alt="Multi-LLM"/>
</div>

<hr>

<p align="center">
  <a href="README.md">English</a> &nbsp;&bull;&nbsp;
  <a href="README.zh-CN.md">简体中文</a> &nbsp;&bull;&nbsp;
  <a href="README.ja.md">日本語</a>
</p>

<p align="center">
  <a href="#デモ">デモ</a> &nbsp;&bull;&nbsp;
  <a href="#機能">機能</a> &nbsp;&bull;&nbsp;
  <a href="#アーキテクチャ">アーキテクチャ</a> &nbsp;&bull;&nbsp;
  <a href="#ベンチマーク">ベンチマーク</a> &nbsp;&bull;&nbsp;
  <a href="#クイックスタート">クイックスタート</a> &nbsp;&bull;&nbsp;
  <a href="#設定">設定</a>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=XortexAI/XMem&type=date&theme=dark&legend=top-left" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=XortexAI/XMem&type=date&legend=top-left" />
  <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=XortexAI/XMem&type=date&legend=top-left" />
</picture>

## 更新 / ニュース
- **[2026 年 6 月 1 日]** XMem のメモリレイヤーにネイティブ Golang 実装が追加されました。より高いスループット、低いレイテンシ、そして数百万回規模のインタラクションでも信頼して動作する本番環境向けデプロイを目的に構築されています。
- **[2026 年 5 月 25 日]** ローカルワークスペース対応が公開されました。わずか 3 コマンドで XMem をローカルにセットアップし、数分でメモリ付きアプリケーションの構築を始められます。セットアップ手順は [Local.md](https://github.com/XortexAI/XMem/blob/main/Local.md) を参照してください。
 ```bash
npx create-xmem@latest
cd xmem
npm run dev
```

## XMem とは？

LLM との会話は毎回ゼロから始まります。ツールを切り替え、プロバイダーを切り替え、翌週に戻ってくると、すべてのコンテキストは失われています。

XMem はインドで No.1 のオープンソース Agentic Memory Layer です。私たちは Memory-as-a-Service、つまりあらゆる AI ユースケースとドメインのためのメモリレイヤーを提供します。長時間稼働するエージェントのための時間記憶、患者コンテキストのための医療記憶、チームやプロジェクトのためのエンタープライズ記憶、そしてコーディングエージェントやワークフローのための開発者記憶に対応します。

これはステートフル AI のための、初のエージェント型メモリレイヤーです。
従来のメモリシステムがチャンクの保存と検索にとどまるのに対し、XMem はメモリを能動的な推論プロセスに変えます。何を覚え、何を更新し、何を忘れるべきかを判断し、それぞれのメモリを適切な専用エージェントとストアへ動的にルーティングします。

## デモ

任意の AI プラットフォームで「X」と入力するだけで、XMem が提供する 4 つのモードを切り替えられます。メモリの保存と検索、既存チャットからのコンテキスト取り込み、インデックス済みリポジトリの利用をシームレスに行えます。

https://github.com/user-attachments/assets/8e3349ab-63c9-4046-821d-ca8097948440

## 機能

### Chrome 拡張

XMem の Chrome 拡張は、ChatGPT、Claude、Gemini、DeepSeek、Perplexity に永続メモリをもたらします。

**リアルタイム検索と注入** - プロンプトを入力している間、XMem はメモリをリアルタイムに検索し、フローティングチップを表示します。ワンクリックで関連コンテキストを入力欄に直接注入できます。

**バックグラウンド自動保存（Xingest）** - 「送信」を押すと、XMem は会話ターンを非同期に取得します。バックグラウンドキューが事実と要約を抽出するため、UI の操作を妨げません。

https://github.com/user-attachments/assets/97793cf9-d247-4d02-9c31-3cc9bbbf89aa

### エージェントプラグイン

新しい [`plugin/`](plugin/) フォルダにより、XMem を開発者向けエージェントやコーディングアシスタントへ直接組み込めます。Claude Code、Codex、Cursor、Hermes、OpenClaw、OpenCode 向けのファーストパーティ統合を含み、エージェントは既存メモリの検索、永続的なプロジェクト知識の保存、セッションをまたいだコンテキスト維持を行えます。API キーは環境変数またはクライアント固有のシークレットストアに保持できます。

### Context

Context を使うと、手動でコピー＆ペーストすることなく既存の会話を XMem に取り込めます。

共有された ChatGPT、Claude、Gemini のリンクを貼り付けると、XMem がそれを開き、すべてのユーザー発言とアシスタント発言を抽出し、完全な ingest pipeline を実行して、その会話を検索可能なメモリにします。

トランスクリプトファイル（テキスト、Markdown、JSON）をアップロードすることもできます。XMem は Cursor と Antigravity のエクスポート形式を標準で解析し、未知の形式には LLM ベースのフォールバック解析を使用します。

https://github.com/user-attachments/assets/4ff22405-b7ad-4b78-9189-9a6e3ebd5e40

### Scanner

Scanner は Git リポジトリ全体をインデックスし、コードベースのクエリ可能なナレッジグラフを構築します。

インデックス後は、ファイル、関数、依存関係、影響範囲について自然言語で質問できます。新しいリポジトリの理解、機能の場所の特定、コードのつながりの追跡、変更によって壊れる可能性がある箇所の把握に役立ちます。

https://github.com/user-attachments/assets/f0fd393e-3820-404b-8d0e-e452e1dd52d0

### マルチドメイン分類

すべてのメモリが同じではありません。それらを同じものとして扱うことが、他のソリューションが伸び悩む理由です。XMem の **Classifier Agent** は、入力されたすべてのデータを分析し、適切なドメインへルーティングします。

<table>
  <tr>
    <th>ドメイン</th>
    <th>保存する内容</th>
    <th>例</th>
    <th>ストレージ</th>
  </tr>
  <tr>
    <td><strong>Profile</strong></td>
    <td>永続的なユーザー事実、好み、アイデンティティ</td>
    <td><em>「バックエンドでは Python より Go が好き」</em></td>
    <td>Pinecone</td>
  </tr>
  <tr>
    <td><strong>Temporal</strong></td>
    <td>日付解決を伴う時間に紐づいたイベント</td>
    <td><em>「昨日 Staff Engineer に昇進した」</em></td>
    <td>Neo4j</td>
  </tr>
  <tr>
    <td><strong>Summary</strong></td>
    <td>圧縮された会話の要点</td>
    <td><em>「REST から gRPC への移行について話した」</em></td>
    <td>Pinecone</td>
  </tr>
  <tr>
    <td><strong>Code</strong></td>
    <td>シンボルに紐づく注釈、バグ、説明</td>
    <td><em>「このリトライロジックには競合状態がある」</em></td>
    <td>Neo4j + Pinecone</td>
  </tr>
  <tr>
    <td><strong>Snippet</strong></td>
    <td>個人のコードパターンやユーティリティ</td>
    <td><em>「これは Go で使う標準的なエラーハンドラ」</em></td>
    <td>Pinecone</td>
  </tr>
  <tr>
    <td><strong>Image</strong></td>
    <td>視覚的な観察と説明</td>
    <td><em>アーキテクチャ図のスクリーンショット</em></td>
    <td>Pinecone</td>
  </tr>
</table>

### Agentic Retrieval

XMem に問い合わせるとき、検索は単純なベクトル検索ではありません。LLM 自身が何を調べるべきかを判断します。

1. **ツール選択** - Retrieval LLM がクエリを分析し、適切な検索ツール（SearchProfile、SearchTemporal、SearchSummary、SearchSnippet）を呼び出します。必要に応じて複数のツールを並列に呼び出します。
2. **統合生成** - すべての検索ツールの結果を集約し、LLM が出典付きの回答を生成します。

つまり「私の好みの技術スタックは何で、auth モジュールを最後にリファクタしたのはいつ？」と聞くと、profile 検索と temporal 検索が自動的に同時実行されます。

### フォールバック付きマルチ LLM オーケストレーション

XMem は 1 つのプロバイダーに縛られません。**Gemini、Claude、OpenAI、OpenRouter、Amazon Bedrock、Ollama** をまたいでオーケストレーションし、自動フェイルオーバーを行います。

```
gemini -> claude -> openai -> bedrock -> ollama
```

メインの LLM がレート制限を受けたり停止したりした場合、XMem は次のプロバイダーへ静かにフォールバックします。各エージェントは特定のモデルに固定でき、フォールバック順序は完全に設定可能です。

### ローカル実行

クラウド依存は不要です。LLM に Ollama、埋め込みに FastEmbed、ベクトルストレージに Chroma または SQLite を使って XMem を実行できます。

```bash
pip install -e ".[local]"
```

## アーキテクチャ

<img width="1536" height="1024" alt="WhatsApp Image 2026-04-27 at 11 50 51" src="https://github.com/user-attachments/assets/424d1c77-63e3-48ac-b457-6beecd437f65" />

XMem は、LangGraph によって調整される**専用 AI エージェントのパイプライン**として構築され、決定論的実行レイヤー（Weaver）と 3 つの目的特化型ストレージエンジンに支えられています。

### 取り込みフロー

```
ユーザー入力（SDK / Chrome 拡張 / API）
         |
         v
   +--------------+
   |  Classifier  |    テキストを分析し、ドメインへルーティング
   +------+-------+
          |
    +-----+-----+------+----------+
    v     v     v      v          v
 Profile Temporal Summary Code  Snippet     ドメインエージェントが構造化データを並列抽出
 Agent   Agent   Agent  Agent   Agent
    |     |      |      |        |
    v     v      v      v        v
   +----------------------------------+
   |          Judge Agent             |     既存メモリと比較
   |   (ADD / UPDATE / DELETE / NOOP) |     重複と陳腐化を防止
   +----------------+-----------------+
                    |
                    v
   +----------------------------------+
   |        Weaver（Rust core）       |     決定論的エグゼキュータ
   |  Pinecone | Neo4j | MongoDB     |     LLM なし。純粋なソフトウェアロジック
   +----------------------------------+
```

1. **Classifier** が入力を関連ドメインへルーティングします。
2. **Domain Agents**（Profiler、Temporal、Summarizer、Code、Snippet、Image）が構造化データを並列抽出します。
3. **Judge Agent** が各抽出結果を既存メモリと比較し、ADD、UPDATE、DELETE、NOOP のいずれかを判断します。
4. **Weaver** が Judge の判断をすべてのストレージバックエンドに対して決定論的に実行します。コアは LLM に依存しない独立した Rust crate として実装されています。

**高 effort モード**では、長い入力を重複するチャンク（約 200 token）に自動分割して並列処理し、結果を統合することで長い会話の取りこぼしを防ぎます。

### 検索フロー

```
ユーザークエリ
    |
    v
+----------------------------------+
|       Retrieval LLM              |
|  呼び出すツールを決定：          |
|  SearchProfile, SearchTemporal,  |
|  SearchSummary, SearchSnippet    |
+----------------+-----------------+
                 |
    +------------+------------+
    v            v            v
 Pinecone      Neo4j      Pinecone        並列検索実行
 (profiles)   (events)   (summaries)
    |            |            |
    +------------+------------+
                 v
+----------------------------------+
|   回答統合 + 引用                |    LLM が出典付き回答を生成
+----------------------------------+
```

### ストレージ

<table>
  <tr>
    <th>エンジン</th>
    <th>目的</th>
    <th>用途</th>
  </tr>
  <tr>
    <td><strong>Pinecone</strong></td>
    <td>高速なベクトル類似検索</td>
    <td>Profile、要約、スニペット、コード注釈</td>
  </tr>
  <tr>
    <td><strong>Neo4j</strong></td>
    <td>グラフ探索 + 時間推論</td>
    <td>イベント、コード知識グラフ、注釈</td>
  </tr>
  <tr>
    <td><strong>MongoDB</strong></td>
    <td>生ドキュメント保存</td>
    <td>スキャン済みコード、ファイルメタデータ、スキャン状態</td>
  </tr>
</table>

> [!NOTE]
> ローカルデプロイでは、Pinecone を **Chroma**、**pgvector**、または **SQLite** ベクトルストアに置き換えられます。

## ベンチマーク

XMem を主要なメモリソリューションとともに、確立された 2 つの学術ベンチマークで評価しました。XMem は全体的に優れた結果を示しています。

### LoCoMo

メモリに対する合成推論をテストします。システムが会話をまたいだ事実を結び付け、時間関係を推論し、自由回答形式の質問に答えられるかを評価します。

<table>
  <tr>
    <th>手法</th>
    <th>Single-Hop (%)</th>
    <th>Multi-Hop (%)</th>
    <th>Open Domain (%)</th>
    <th>Temporal (%)</th>
    <th>Overall (%)</th>
  </tr>
  <tr><td><strong>XMEM（私たちの手法）</strong></td><td><strong>90.6</strong></td><td><strong>92.3</strong></td><td><strong>91.2</strong></td><td><strong>91.9</strong></td><td><strong>91.5</strong></td></tr>
  <tr><td>Zep</td><td>74.11</td><td>66.04</td><td>67.71</td><td>79.79</td><td>75.14</td></tr>
  <tr><td>Memobase (v0.0.37)</td><td>70.92</td><td>46.88</td><td>77.17</td><td>85.05</td><td>75.78</td></tr>
  <tr><td>Mem0g (YC 24)</td><td>65.71</td><td>47.19</td><td>75.71</td><td>58.13</td><td>68.44</td></tr>
  <tr><td>Mem0 (YC 24)</td><td>67.13</td><td>51.15</td><td>72.93</td><td>55.51</td><td>66.88</td></tr>
  <tr><td>LangMem</td><td>62.23</td><td>47.92</td><td>71.12</td><td>23.43</td><td>58.10</td></tr>
  <tr><td>OpenAI</td><td>63.79</td><td>42.92</td><td>62.29</td><td>21.71</td><td>52.90</td></tr>
</table>

> マルチホップ推論（異なる会話の事実を結び付ける能力）では、XMem は次点のシステムを **26.3 ポイント** 上回ります。総合でも XMem は **91.5%** で全システムをリードし、Zep の 75.14 を上回っています。

### LongMemEval-S

長期会話メモリの業界標準ベンチマークです。事実の想起、好みの変化の追跡、時間推論、セッションをまたいだコンテキスト保持を評価します。

<table>
  <tr>
    <th>カテゴリ</th>
    <th>XMem (Gemini 3-flash)</th>
    <th>Backboard.io (GPT-4o)</th>
    <th>Mastra (GPT-4o)</th>
    <th>Supermemory (GPT-4o)</th>
  </tr>
  <tr><td><strong>マルチセッション</strong></td><td><strong>93.6</strong></td><td>91.7</td><td>79.7</td><td>71.43</td></tr>
  <tr><td><strong>時間推論</strong></td><td><strong>94.5</strong></td><td>91.7</td><td>85.7</td><td>76.69</td></tr>
  <tr><td><strong>単一セッション（アシスタント）</strong></td><td><strong>96.43</strong></td><td>98.2</td><td>82.1</td><td>96.43</td></tr>
  <tr><td><strong>単一セッション（ユーザー）</strong></td><td><strong>97.1</strong></td><td>97.1</td><td>98.6</td><td>97.14</td></tr>
  <tr><td><strong>知識更新</strong></td><td><strong>91.2</strong></td><td>93.6</td><td>85.9</td><td>88.46</td></tr>
  <tr><td><strong>単一セッションの好み</strong></td><td><strong>87.0</strong></td><td>90.0</td><td>73.3</td><td>70.0</td></tr>
</table>

> XMem はすべてのカテゴリで Backboard.io に匹敵し、セッション想起と好みの追跡でほぼ満点を記録しています。総合では Mastra を **9.2 ポイント**、Supermemory を **11.8 ポイント** 上回ります。

### ベンチマーク方法
- **評価**：構造化ルーブリックを用いた Gemini による LLM-as-Judge
- **公平性**：すべてのシステムを同一の会話履歴とクエリでテスト

## クイックスタート

### ローカル XMem

```bash
npx create-xmem@latest
cd xmem
npm run dev
```

Windows、macOS、Linux で動作します。ローカル XMem ワークスペースを作成し、バックエンドをインストールし、ローカルストレージを起動し、Chrome 拡張をビルドし、`http://localhost:8000` で API を起動します。

ローカル前提条件：

- Git
- Node.js 20+
- Python 3.11+
- Docker Desktop
- `.env` にクラウド LLM key を追加しない場合は Ollama

セットアップ後、次の場所から拡張機能を読み込みます。

```text
repos/xmem-extension/dist
```

Chrome のパス：`chrome://extensions` -> デベロッパーモードを有効化 -> パッケージ化されていない拡張機能を読み込む。

### ローカルコマンド

```bash
npm run setup
npm run start
npm run verify
npm run doctor
```

`.env` に実際のクラウド LLM key が含まれている場合、XMem はそのプロバイダーを使用し、埋め込みは FastEmbed でローカルに保ちます。クラウド key が設定されていない場合、XMem はローカル Ollama にフォールバックし、セットアップ中に必要なローカルモデルを取得します。

### コンテキストのポータビリティ

```bash
npm run context:export
npm run context:import -- --file ./exports/xmem-context.json
npm run context:sync -- --file ./exports/xmem-context.json --server https://api.xmem.in --api-key <key>
```

`context:export` は、後でインポートまたは XMem サーバーへ同期できるローカルコンテキストバンドルを書き出します。

### リポジトリをインデックスする

```bash
python -m src.scanner.runner \
  --org your-org \
  --repo your-repo \
  --url https://github.com/your-org/your-repo.git \
  --enrich
```

> [!TIP]
> クラウド依存のない完全ローカルセットアップの場合：
> ```ini
> FALLBACK_ORDER='["ollama"]'
> EMBEDDING_PROVIDER=ollama
> VECTOR_STORE_PROVIDER=pgvector
> ```
> その後、ローカル追加依存をインストールします：`pip install -e ".[local]"`

## 設定

XMem は高度に設定可能です。任意のエージェントのモデルを上書きし、フォールバックチェーンを調整し、品質と速度のトレードオフを変更できます。

<table>
  <tr>
    <th>設定</th>
    <th>デフォルト</th>
    <th>説明</th>
  </tr>
  <tr><td><code>FALLBACK_ORDER</code></td><td><code>openrouter,gemini,claude,openai</code></td><td>プロバイダーのフェイルオーバー順序</td></tr>
  <tr><td><code>DEEPSEEK_API_KEY</code></td><td>empty</td><td>公式 OpenAI 互換エンドポイント用の DeepSeek API key</td></tr>
  <tr><td><code>MIMO_API_KEY</code></td><td>empty</td><td>公式 OpenAI 互換エンドポイント用の Xiaomi MiMo API key</td></tr>
  <tr><td><code>CLASSIFIER_MODEL</code></td><td>default model</td><td>classifier agent のモデルを上書き</td></tr>
  <tr><td><code>JUDGE_MODEL</code></td><td>default model</td><td>judge agent のモデルを上書き</td></tr>
  <tr><td><code>RETRIEVAL_MODEL</code></td><td>default model</td><td>検索統合モデルを上書き</td></tr>
  <tr><td><code>EMBEDDING_MODEL</code></td><td><code>gemini-embedding-001</code></td><td>テキスト埋め込みモデル</td></tr>
  <tr><td><code>EMBEDDING_PROVIDER</code></td><td><code>auto</code></td><td>auto, gemini, bedrock, ollama, fastembed</td></tr>
  <tr><td><code>VECTOR_STORE_PROVIDER</code></td><td><code>pinecone</code></td><td>pinecone, pgvector, chroma, sqlite</td></tr>
  <tr><td><code>PINECONE_DIMENSION</code></td><td><code>768</code></td><td>埋め込みベクトルの次元数</td></tr>
  <tr><td><code>RATE_LIMIT</code></td><td><code>60</code></td><td>1 分あたりの API リクエスト数</td></tr>
  <tr><td><code>TEMPERATURE</code></td><td><code>0.4</code></td><td>LLM 生成温度</td></tr>
</table>
