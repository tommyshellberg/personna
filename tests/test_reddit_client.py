"""Tests for RedditClient including submission and comment fetching."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.reddit_client import RedditClient


class TestParseSubmissionId:
    """Tests for extracting submission ID from Reddit URLs."""

    @pytest.fixture
    def reddit_config(self):
        """Minimal config for RedditClient."""
        return {
            "rate_limit_seconds": 5,
            "max_comments_per_user": 100
        }

    @pytest.fixture
    def client(self, reddit_config):
        """RedditClient with mocked PRAW."""
        with patch('src.reddit_client.praw.Reddit'):
            return RedditClient(reddit_config)

    def test_parses_standard_url(self, client):
        """Standard reddit.com URL with subreddit and title."""
        url = "https://www.reddit.com/r/productivity/comments/abc123/my_post_title/"

        result = client._parse_submission_id(url)

        assert result == "abc123"

    def test_parses_url_without_www(self, client):
        """URL without www prefix."""
        url = "https://reddit.com/r/startups/comments/xyz789/another_post/"

        result = client._parse_submission_id(url)

        assert result == "xyz789"

    def test_parses_old_reddit_url(self, client):
        """Old Reddit subdomain URL."""
        url = "https://old.reddit.com/r/Python/comments/def456/python_tips/"

        result = client._parse_submission_id(url)

        assert result == "def456"

    def test_parses_short_url(self, client):
        """Short redd.it URL format."""
        url = "https://redd.it/abc123"

        result = client._parse_submission_id(url)

        assert result == "abc123"

    def test_parses_url_without_trailing_slash(self, client):
        """URL without trailing slash."""
        url = "https://www.reddit.com/r/test/comments/test123/title"

        result = client._parse_submission_id(url)

        assert result == "test123"

    def test_parses_url_with_query_params(self, client):
        """URL with query parameters."""
        url = "https://www.reddit.com/r/test/comments/qp123/title/?utm_source=share"

        result = client._parse_submission_id(url)

        assert result == "qp123"

    def test_raises_on_invalid_url(self, client):
        """Invalid URL raises ValueError."""
        url = "https://example.com/not-reddit"

        with pytest.raises(ValueError, match="Could not parse submission ID"):
            client._parse_submission_id(url)

    def test_raises_on_user_profile_url(self, client):
        """User profile URL (not a post) raises ValueError."""
        url = "https://www.reddit.com/user/someuser/comments/"

        with pytest.raises(ValueError, match="Could not parse submission ID"):
            client._parse_submission_id(url)


class TestGetSubmission:
    """Tests for fetching submission metadata."""

    @pytest.fixture
    def reddit_config(self):
        return {"rate_limit_seconds": 5, "max_comments_per_user": 100}

    @pytest.fixture
    def mock_submission(self):
        """Mock PRAW submission object."""
        submission = MagicMock()
        submission.id = "abc123"
        submission.title = "I built an app that tracks habits"
        submission.selftext = "Here's my new productivity app..."
        submission.subreddit = MagicMock()
        submission.subreddit.__str__ = lambda x: "productivity"
        submission.score = 150
        return submission

    @pytest.fixture
    def client_with_mock(self, reddit_config, mock_submission):
        """RedditClient with mocked PRAW that returns our mock submission."""
        with patch('src.reddit_client.praw.Reddit') as mock_reddit_class:
            mock_reddit = MagicMock()
            mock_reddit.submission.return_value = mock_submission
            mock_reddit_class.return_value = mock_reddit
            client = RedditClient(reddit_config)
            return client

    def test_returns_submission_metadata(self, client_with_mock, mock_submission):
        """get_submission returns dict with post metadata."""
        url = "https://www.reddit.com/r/productivity/comments/abc123/my_app/"

        result = client_with_mock.get_submission(url)

        assert result['id'] == "abc123"
        assert result['title'] == "I built an app that tracks habits"
        assert result['selftext'] == "Here's my new productivity app..."
        assert result['subreddit'] == "productivity"
        assert result['score'] == 150
        assert result['url'] == url

    def test_uses_parsed_submission_id(self, client_with_mock):
        """Verifies it extracts ID from URL and calls PRAW correctly."""
        url = "https://reddit.com/r/test/comments/xyz789/title/"

        client_with_mock.get_submission(url)

        client_with_mock.reddit.submission.assert_called_once_with(id="xyz789")


class TestGetTopLevelComments:
    """Tests for fetching top-level comments from a submission."""

    @pytest.fixture
    def reddit_config(self):
        return {"rate_limit_seconds": 5, "max_comments_per_user": 100}

    @pytest.fixture
    def mock_comments(self):
        """Mock PRAW comment objects."""
        comment1 = MagicMock()
        comment1.id = "c1"
        comment1.author = MagicMock()
        comment1.author.__str__ = lambda x: "user1"
        comment1.body = "This is exactly what I needed!"
        comment1.score = 42
        comment1.created_utc = 1700000000
        comment1.permalink = "/r/productivity/comments/abc123/title/c1/"

        comment2 = MagicMock()
        comment2.id = "c2"
        comment2.author = MagicMock()
        comment2.author.__str__ = lambda x: "user2"
        comment2.body = "Meh, seen this before"
        comment2.score = -3
        comment2.created_utc = 1700001000
        comment2.permalink = "/r/productivity/comments/abc123/title/c2/"

        return [comment1, comment2]

    @pytest.fixture
    def mock_submission_with_comments(self, mock_comments):
        """Mock submission with comments."""
        submission = MagicMock()
        submission.comments = MagicMock()
        submission.comments.replace_more = MagicMock()
        submission.comments.__iter__ = lambda x: iter(mock_comments)
        return submission

    @pytest.fixture
    def client_with_mock(self, reddit_config, mock_submission_with_comments):
        """RedditClient with mocked PRAW."""
        with patch('src.reddit_client.praw.Reddit') as mock_reddit_class:
            mock_reddit = MagicMock()
            mock_reddit.submission.return_value = mock_submission_with_comments
            mock_reddit_class.return_value = mock_reddit
            client = RedditClient(reddit_config)
            return client

    def test_returns_list_of_comment_dicts(self, client_with_mock):
        """get_top_level_comments returns list of comment dictionaries."""
        url = "https://www.reddit.com/r/productivity/comments/abc123/title/"

        result = client_with_mock.get_top_level_comments(url)

        assert len(result) == 2
        assert result[0]['id'] == "c1"
        assert result[0]['author'] == "user1"
        assert result[0]['body'] == "This is exactly what I needed!"
        assert result[0]['score'] == 42
        assert result[0]['created_utc'] == 1700000000
        assert "reddit.com" in result[0]['permalink']

    def test_calls_replace_more_to_skip_nested(self, client_with_mock, mock_submission_with_comments):
        """Verifies replace_more(limit=0) is called to skip 'load more' links."""
        url = "https://www.reddit.com/r/test/comments/abc123/title/"

        client_with_mock.get_top_level_comments(url)

        mock_submission_with_comments.comments.replace_more.assert_called_once_with(limit=0)

    def test_handles_deleted_author(self, reddit_config):
        """Comments from deleted users have '[deleted]' as author."""
        comment = MagicMock()
        comment.id = "c1"
        comment.author = None  # Deleted user
        comment.body = "Some comment"
        comment.score = 10
        comment.created_utc = 1700000000
        comment.permalink = "/r/test/comments/abc123/title/c1/"

        submission = MagicMock()
        submission.comments = MagicMock()
        submission.comments.replace_more = MagicMock()
        submission.comments.__iter__ = lambda x: iter([comment])

        with patch('src.reddit_client.praw.Reddit') as mock_reddit_class:
            mock_reddit = MagicMock()
            mock_reddit.submission.return_value = submission
            mock_reddit_class.return_value = mock_reddit
            client = RedditClient(reddit_config)

            result = client.get_top_level_comments("https://reddit.com/r/test/comments/abc123/")

            assert result[0]['author'] == "[deleted]"
