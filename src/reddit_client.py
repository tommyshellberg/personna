import praw
import os
import re
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

    def _parse_submission_id(self, url: str) -> str:
        """Extract submission ID from various Reddit URL formats.

        Supported formats:
        - https://www.reddit.com/r/subreddit/comments/abc123/title/
        - https://reddit.com/r/subreddit/comments/abc123/title
        - https://old.reddit.com/r/subreddit/comments/abc123/title
        - https://redd.it/abc123

        Args:
            url: Full Reddit post URL

        Returns:
            Submission ID string

        Raises:
            ValueError: If URL format is not recognized
        """
        # Pattern for standard Reddit URLs: /comments/SUBMISSION_ID/
        standard_pattern = r'/comments/([a-zA-Z0-9]+)'
        match = re.search(standard_pattern, url)
        if match:
            return match.group(1)

        # Pattern for short redd.it URLs: redd.it/SUBMISSION_ID
        short_pattern = r'redd\.it/([a-zA-Z0-9]+)'
        match = re.search(short_pattern, url)
        if match:
            return match.group(1)

        raise ValueError(f"Could not parse submission ID from URL: {url}")

    def get_submission(self, post_url: str) -> Dict[str, Any]:
        """Fetch post metadata from URL.

        Args:
            post_url: Full Reddit post URL

        Returns:
            Dictionary with submission metadata:
                - id: Submission ID
                - title: Post title
                - selftext: Post body text
                - subreddit: Subreddit name
                - score: Reddit score
                - url: Original URL
        """
        submission_id = self._parse_submission_id(post_url)
        submission = self.reddit.submission(id=submission_id)

        return {
            'id': submission.id,
            'title': submission.title,
            'selftext': submission.selftext,
            'subreddit': str(submission.subreddit),
            'score': submission.score,
            'url': post_url
        }

    def get_top_level_comments(self, post_url: str) -> List[Dict[str, Any]]:
        """Fetch top-level comments only (no nested replies).

        Args:
            post_url: Full Reddit post URL

        Returns:
            List of comment dictionaries with:
                - id: Comment ID
                - author: Username or '[deleted]'
                - body: Comment text
                - score: Reddit score
                - created_utc: Unix timestamp
                - permalink: Full Reddit URL
        """
        submission_id = self._parse_submission_id(post_url)
        submission = self.reddit.submission(id=submission_id)

        # Skip "load more comments" links - just get top-level
        submission.comments.replace_more(limit=0)

        comments = []
        for comment in submission.comments:
            # Skip MoreComments objects that might slip through
            if not hasattr(comment, 'body'):
                continue

            author = str(comment.author) if comment.author else '[deleted]'

            comments.append({
                'id': comment.id,
                'author': author,
                'body': comment.body,
                'score': comment.score,
                'created_utc': comment.created_utc,
                'permalink': f"https://reddit.com{comment.permalink}"
            })

        return comments