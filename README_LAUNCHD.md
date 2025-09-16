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
/Users/arigatatsuya/Work/git/nook/logs/                # ログディレクトリ
├── datacollection.log                                 # 標準出力ログ
└── datacollection_error.log                          # エラーログ
```

### 環境設定
- **AI Client**: Claude CLI
- **Python**: /Users/arigatatsuya/.pyenv/versions/3.12.4/bin/python
- **出力ディレクトリ**: /Users/arigatatsuya/Library/Mobile Documents/iCloud~md~obsidian/Documents/obsidian-iphone/nook

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
# 標準出力ログ
tail -f /Users/arigatatsuya/Work/git/nook/logs/datacollection.log

# エラーログ
tail -f /Users/arigatatsuya/Work/git/nook/logs/datacollection_error.log
```

## トラブルシューティング

### よくある問題
1. **Claude CLIが見つからない**: PATHに`/opt/homebrew/bin`が含まれているか確認
2. **権限エラー**: plistファイルの権限が644になっているか確認
3. **Python環境エラー**: pyenvのPythonパスが正しいか確認

### 設定の再適用
```bash
cd /Users/arigatatsuya/Work/git/nook
launchctl unload ~/Library/LaunchAgents/com.nook.datacollection.plist
cp com.nook.datacollection.plist ~/Library/LaunchAgents/
chmod 644 ~/Library/LaunchAgents/com.nook.datacollection.plist
launchctl load ~/Library/LaunchAgents/com.nook.datacollection.plist
```

## 出力確認
データ収集の結果は以下のディレクトリに保存されます：
```
/Users/arigatatsuya/Library/Mobile Documents/iCloud~md~obsidian/Documents/obsidian-iphone/nook/
├── github_trending/
├── hacker_news/
├── paper_summarizer/
└── tech_feed/
```

これらのファイルはiCloud経由でObsidianアプリと自動同期されます。

