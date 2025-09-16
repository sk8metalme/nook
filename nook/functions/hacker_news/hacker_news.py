import inspect
import os
import traceback
from dataclasses import dataclass
from datetime import date
from pprint import pprint
from typing import Any

import requests
from bs4 import BeautifulSoup
from ..common.python.client_factory import create_client

_MARKDOWN_FORMAT = """
# {title}

**Score**: {score}

{url_or_text}

**日本語タイトル**: {japanese_title}

**記事要約**: {article_summary}
"""

class Config:
    hacker_news_top_stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    hacker_news_item_url = "https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
    hacker_news_num_top_stories = 30
    summary_index_s3_key_format = "hacker_news/{date}.md"

@dataclass
class Story:
    title: str
    score: int
    url: str | None = None
    text: str | None = None

class HackerNewsRetriever:
    def __init__(self):
        self._client = create_client()

    def __call__(self) -> None:
        stories = self._get_top_stories()
        styled_attachments = [self._stylize_story(story) for story in stories]
        self._store_summaries(styled_attachments)

    def _get_top_stories(self) -> list[Story]:
        top_stories = self._get_top_storie_ids()[:Config.hacker_news_num_top_stories]
        stories = []
        for story_id in top_stories:
            story = self._get_story(story_id)
            if story["score"] < 20:
                continue
            summary = None
            if story.get("text"):
                if 100 < len(story["text"]) < 10000:
                    summary = self._summarize_story(story)
                else:
                    summary = self._cleanse_text(story["text"])
            stories.append(
                Story(
                    title=story["title"],
                    score=story["score"],
                    url=story.get("url"),
                    text=story.get("text") if summary is None else summary,
                )
            )
        return stories

    def _summarize_story(self, story: dict[str, str | int]) -> str:
        return self._client.generate_content(
            contents=self._contents_format.format(
                title=story["title"], text=self._cleanse_text(story["text"])
            ),
            system_instruction=self._system_instruction,
        )

    def _get_top_storie_ids(self) -> list[int]:
        return requests.get(
            Config.hacker_news_top_stories_url,
            headers={"Content-Type": "application/json"},
        ).json()

    def _get_story(self, story_id: int) -> dict[str, str]:
        return requests.get(
            Config.hacker_news_item_url.format(story_id=story_id),
            headers={"Content-Type": "application/json"},
        ).json()

    def _cleanse_text(self, text: str) -> str:
        return BeautifulSoup(text, "html.parser").get_text()

    def _store_summaries(self, summaries: list[str]) -> None:
        date_str = date.today().strftime("%Y-%m-%d")
        key = Config.summary_index_s3_key_format.format(date=date_str)
        output_dir = os.environ.get("OUTPUT_DIR", "./output")
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, key)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n---\n".join(summaries))
        print(f"Saved summaries to {file_path}")

    def _stylize_story(self, story: Story) -> str:
        url_or_text = f"[View Link]({story.url})" if story.url else story.text
        # タイトルを日本語に翻訳
        japanese_title = self._translate_title_to_japanese(story.title)
        
        # 記事の要約を生成
        article_summary = self._get_article_summary(story)
        
        return _MARKDOWN_FORMAT.format(
            title=story.title,
            score=story.score,
            url_or_text=url_or_text,
            japanese_title=japanese_title,
            article_summary=article_summary,
        )
    
    def _translate_title_to_japanese(self, title: str) -> str:
        """英語のタイトルを日本語に翻訳する"""
        try:
            prompt = f"""以下の英語のニュースタイトルを自然な日本語に翻訳してください。技術用語や固有名詞は適切に翻訳し、ニュースタイトルとして自然な日本語にしてください。

英語タイトル: {title}

日本語タイトル:"""
            
            response = self._client.generate_content(prompt)
            return response.strip()
        except Exception as e:
            print(f"タイトル翻訳エラー: {e}")
            return title  # 翻訳に失敗した場合は元のタイトルを返す
    
    def _get_article_summary(self, story: Story) -> str:
        """記事の内容を取得し、日本語で要約する"""
        try:
            if not story.url:
                # URLがない場合はHacker Newsの投稿テキストを要約
                if story.text and len(story.text) > 50:
                    return self._summarize_text_content(story.text, story.title)
                else:
                    return "要約できる内容がありません"
            
            # URLがある場合は記事を取得して要約
            return self._fetch_and_summarize_article(story.url, story.title)
            
        except Exception as e:
            print(f"記事要約エラー: {e}")
            return "記事要約を生成できませんでした"
    
    def _fetch_and_summarize_article(self, url: str, title: str) -> str:
        """URLから記事を取得し、日本語で要約する"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return "記事を取得できませんでした"
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 記事本文を抽出（一般的なタグを試行）
            content = ""
            for selector in ['article', 'main', '.content', '.post-content', '.entry-content', 'p']:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text().strip() for elem in elements])
                    break
            
            if not content:
                # フォールバック: すべてのpタグを取得
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs])
            
            if len(content) < 100:
                return "記事内容を十分に取得できませんでした"
            
            # 長すぎる場合は制限
            if len(content) > 3000:
                content = content[:3000] + "..."
            
            return self._summarize_text_content(content, title)
            
        except Exception as e:
            print(f"記事取得エラー ({url}): {e}")
            return "記事の取得に失敗しました"
    
    def _summarize_text_content(self, content: str, title: str) -> str:
        """テキスト内容を日本語で要約する"""
        try:
            prompt = f"""以下のニュース記事を日本語で簡潔に要約してください。
- 記事の主要なポイント
- 重要な事実や数字
- 影響や意義
を含めて、3-4文程度で要約してください。

タイトル: {title}

記事内容:
{content}

日本語要約:"""
            
            response = self._client.generate_content(prompt)
            return response.strip()
            
        except Exception as e:
            print(f"テキスト要約エラー: {e}")
            return "要約の生成に失敗しました"

    @property
    def _system_instruction(self) -> str:
        return inspect.cleandoc(
            """
            あなたは、Hacker Newsの最新の記事を要約するアシスタントです。
            ユーザーからHacker Newsの記事のタイトルと本文を与えられるので、あなたはその記事を日本語で要約してください。
            なお、要約以外の出力は不要です。
            """
        )

    @property
    def _contents_format(self) -> str:
        return inspect.cleandoc(
            """
            タイトル
            ```
            {title}
            ```

            本文
            ```
            {text}
            ```
            """
        )


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    pprint(event)

    try:
        # if the lambda is invoked by a cron job,
        # call the paper summarizer without any incoming text
        if event.get("source") == "aws.events":
            retriever = HackerNewsRetriever()
            retriever()

        return {"statusCode": 200}
    except Exception as e:
        pprint(traceback.format_exc())
        pprint(e)
        return {"statusCode": 500}
