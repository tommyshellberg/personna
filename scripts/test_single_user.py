#!/usr/bin/env python3
"""Test fetching comments for a single user"""

import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from src.reddit_client import RedditClient

load_dotenv()

def test_single_user():
    # Load config
    with open('config/settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)
    
    # Test with a public Reddit user
    test_username = "spez"  # Reddit CEO - safe public account
    
    print(f"Testing comment fetch for u/{test_username}...")
    
    try:
        client = RedditClient(settings['reddit'])
        comments = client.get_user_comments(test_username)
        
        print(f"✓ Fetched {len(comments)} comments!")
        print(f"\nFirst few comments:")
        
        for i, comment in enumerate(comments[:3]):
            print(f"\n{i+1}. r/{comment['subreddit']} (score: {comment['score']})")
            print(f"   {comment['body'][:100]}...")
        
        # Test markdown export
        output_dir = Path("data/output")
        output_dir.mkdir(exist_ok=True)
        
        markdown_path = output_dir / f"{test_username}_test.md"
        client.save_comments_to_markdown(comments, test_username, markdown_path)
        
        print(f"\n✓ Saved markdown to: {markdown_path}")
        print(f"✓ File size: {markdown_path.stat().st_size} bytes")
        
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_single_user()