# main.py
import os
from datetime import date
from nook.functions.paper_summarizer.paper_summarizer import PaperSummarizer
from nook.functions.hacker_news.hacker_news import HackerNewsRetriever
from nook.functions.reddit_explorer.reddit_explorer import RedditExplorer
from nook.functions.github_trending.github_trending import GithubTrending
from nook.functions.tech_feed.tech_feed import TechFeed
from dotenv import load_dotenv

def run_all():
    # 環境変数の読み込み
    load_dotenv()
    
    # 出力ディレクトリを設定（環境変数がなければデフォルト）
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["OUTPUT_DIR"] = OUTPUT_DIR

    # 各情報源の処理を実行（Reddit APIキーが未設定のためRedditExplorerは除外）
    handlers = [
        PaperSummarizer(),
        HackerNewsRetriever(),
        # RedditExplorer(),  # Reddit API設定が必要
        GithubTrending(),
        TechFeed(),
    ]
    
    for handler in handlers:
        print(f"Running {handler.__class__.__name__}...")
        handler()
        print(f"Completed {handler.__class__.__name__}")

if __name__ == "__main__":
    run_all()