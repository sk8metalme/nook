# Nook launchd自動実行設定

## 概要
macOSのlaunchdを使用して、毎日0時にNookのデータ収集を自動実行するように設定されています。

## 設定内容

### 実行スケジュール
- **実行時刻**: 毎日0時0分
- **実行内容**: 全データソース（GitHub Trending、Hacker News、Paper Summarizer、Tech Feed）からのデータ収集
- **出力先**: Obsidianディレクトリ（iCloud同期対応）

### ファイル構成
```
~/Library/LaunchAgents/com.nook.datacollection.plist  # launchd設定ファイル
$HOME/Work/git/nook/logs/                             # ログディレクトリ（デフォルト値）
├── datacollection.log                                 # 標準出力ログ
└── datacollection_error.log                          # エラーログ
```

### 環境設定
- **AI Client**: Claude CLI（`AI_CLIENT_TYPE`で上書き可能）
- **Python**: デフォルトは`python3`（`NOOK_PYTHON_BIN`で変更可能）
- **リポジトリディレクトリ**: デフォルトは`$HOME/Work/git/nook`（`NOOK_REPO_DIR`で変更可能）
- **Obsidian出力先**: デフォルトは`$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/obsidian-iphone/nook`（`NOOK_OBSIDIAN_DIR`で変更可能）
- **ログディレクトリ**: デフォルトは`$HOME/Work/git/nook/logs`（`NOOK_LOG_DIR`で変更可能）

### カスタマイズ方法

#### 1. `.env`ファイルによる設定（推奨）
プロジェクトルートに`.env`ファイルを作成して環境変数を設定できます。

```bash
# .env.exampleをコピーして.envファイルを作成
cp .env.example .env

# .envファイルを編集して設定をカスタマイズ
vim .env
```

`.env`ファイルで設定可能な変数：

| 変数名 | 役割 | デフォルト値 |
| --- | --- | --- |
| `NOOK_REPO_DIR` | リポジトリの場所 | `$HOME/Work/git/nook` |
| `NOOK_OBSIDIAN_DIR` | Obsidian出力先 | `$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/obsidian-iphone/nook` |
| `NOOK_LOG_DIR` | ログ出力先 | `$HOME/Work/git/nook/logs` |
| `NOOK_PYTHON_BIN` | 使用するPythonバイナリ | `python3` |
| `NOOK_PATH_OVERRIDE` | 実行時の`PATH` | `/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin` |
| `AI_CLIENT_TYPE` | AIクライアントタイプ | `claude` |
| `OUTPUT_DIR` | 出力ディレクトリ | Obsidian出力先と同じ |

#### 2. plistファイルでの直接設定
`.env`ファイルが存在しない場合や、特定の設定を上書きしたい場合は、`com.nook.datacollection.plist`内の`EnvironmentVariables`セクションで直接設定できます。

**設定の優先順位**: `.env`ファイル > plistの環境変数 > デフォルト値

#### 3. Python環境の柔軟な対応
システムは以下の順序でPythonを検索します：

1. **環境変数指定**: `NOOK_PYTHON_BIN`で明示的に指定
2. **システムデフォルト**: `python3`コマンド（PATH上で最初に見つかるもの）

**対応する環境例**:
- **pyenv**: `python3`が自動的にpyenvのアクティブバージョンを使用
- **Homebrew**: `/opt/homebrew/bin/python3`
- **システム標準**: `/usr/bin/python3`
- **仮想環境**: アクティブな仮想環境の`python3`

**特定バージョンを使用したい場合**:
```bash
# .envファイルで指定
NOOK_PYTHON_BIN=/usr/local/bin/python3.11
# または
NOOK_PYTHON_BIN=$HOME/.pyenv/versions/3.12.4/bin/python
```
## 管理コマンド

### ジョブの状態確認
```bash
launchctl list | grep com.nook.datacollection
```

### ジョブの手動実行
```bash
launchctl start com.nook.datacollection
```

### ジョブの停止
```bash
launchctl unload ~/Library/LaunchAgents/com.nook.datacollection.plist
```

### ジョブの再開
```bash
launchctl load ~/Library/LaunchAgents/com.nook.datacollection.plist
```

### ログの確認
```bash
LOG_DIR="${NOOK_LOG_DIR:-$HOME/Work/git/nook/logs}"

# 標準出力ログ
tail -f "$LOG_DIR/datacollection.log"

# エラーログ
tail -f "$LOG_DIR/datacollection_error.log"
```

## トラブルシューティング

### よくある問題
1. **Claude CLIが見つからない**: PATHに`/opt/homebrew/bin`が含まれているか確認
2. **権限エラー**: plistファイルの権限が644になっているか確認
3. **Python環境エラー**: pyenvのPythonパスが正しいか確認

### 設定の再適用
```bash
cd "${NOOK_REPO_DIR:-$HOME/Work/git/nook}"
launchctl unload ~/Library/LaunchAgents/com.nook.datacollection.plist
cp com.nook.datacollection.plist ~/Library/LaunchAgents/
chmod 644 ~/Library/LaunchAgents/com.nook.datacollection.plist
launchctl load ~/Library/LaunchAgents/com.nook.datacollection.plist
```

## 出力確認
データ収集の結果は、通常`NOOK_OBSIDIAN_DIR`で指定したディレクトリに保存されます（未設定の場合はデフォルト値）。例：
```
$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/obsidian-iphone/nook/
├── github_trending/
├── hacker_news/
├── paper_summarizer/
└── tech_feed/
```

これらのファイルはiCloud経由でObsidianアプリと自動同期されます。
