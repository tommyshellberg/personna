import praw
import os
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path


class RedditClient:
    """Handles Reddit API interactions using PRAW."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reddit = self._initialize_reddit()
    
    def _initialize_reddit(self) -> praw.Reddit:
        """Initialize PRAW Reddit client with credentials from environment."""
        return praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT'),
            ratelimit_seconds=self.config.get('rate_limit_seconds', 5)
        )
    
    def get_user_comments(self, username: str) -> List[Dict[str, Any]]:
        """Fetch top comments for a given username."""
        try:
            user = self.reddit.redditor(username)
            comments = []
            
            max_comments = self.config.get('max_comments_per_user', 100)
            
            for comment in user.comments.top(limit=max_comments):
                comment_data = {
                    'body': comment.body,
                    'score': comment.score,
                    'subreddit': str(comment.subreddit),
                    'created_utc': comment.created_utc,
                    'permalink': f"https://reddit.com{comment.permalink}",
                    'parent_type': 'post' if comment.is_root else 'comment'
                }
                comments.append(comment_data)
            
            return comments
            
        except Exception as e:
            raise Exception(f"Failed to fetch comments for {username}: {e}")
    
    def save_comments_to_markdown(self, comments: List[Dict[str, Any]], 
                                username: str, output_path: Path) -> None:
        """Save user comments to markdown file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Reddit Comments Analysis: u/{username}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Comments:** {len(comments)}\n\n")
            
            # Group comments by subreddit
            subreddit_groups = {}
            for comment in comments:
                subreddit = comment['subreddit']
                if subreddit not in subreddit_groups:
                    subreddit_groups[subreddit] = []
                subreddit_groups[subreddit].append(comment)
            
            for subreddit, sub_comments in subreddit_groups.items():
                f.write(f"## r/{subreddit} ({len(sub_comments)} comments)\n\n")
                
                for comment in sub_comments:
                    created_date = datetime.fromtimestamp(comment['created_utc'])
                    f.write(f"### Comment (Score: {comment['score']})\n")
                    f.write(f"**Date:** {created_date.strftime('%Y-%m-%d')}\n")
                    f.write(f"**Link:** [View on Reddit]({comment['permalink']})\n\n")
                    f.write(f"{comment['body']}\n\n")
                    f.write("---\n\n")