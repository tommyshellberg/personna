#!/usr/bin/env python3
"""Test Reddit API authentication"""

import os
import praw
from dotenv import load_dotenv

load_dotenv()

def test_reddit_auth():
    """Test if Reddit credentials work"""
    try:
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT'),
            ratelimit_seconds=5
        )
        
        # Test with a simple API call
        print("Testing Reddit connection...")
        user = reddit.user.me()
        print(f"✓ Connected! Auth user: {user}")
        
        # Test fetching a known user's basic info
        test_user = reddit.redditor("spez")  # Reddit CEO
        print(f"✓ Can fetch user data! Test user created: {test_user.created_utc}")
        
        print("✓ Reddit authentication successful!")
        
    except Exception as e:
        print(f"✗ Reddit authentication failed: {e}")
        print("\nCheck your .env file has:")
        print("REDDIT_CLIENT_ID=your_id_here")
        print("REDDIT_CLIENT_SECRET=your_secret_here") 
        print("REDDIT_USER_AGENT=YourApp/1.0 by YourUsername")

if __name__ == "__main__":
    test_reddit_auth()