"""Sentiment analysis for Reddit comments using Ollama."""

import json
import os
import re
import ollama
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

# Langsmith tracing - optional, only active if LANGCHAIN_TRACING_V2=true
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    # Fallback decorator that does nothing
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    LANGSMITH_AVAILABLE = False


@dataclass
class SentimentResult:
    """Result of sentiment analysis for a single comment."""
    comment_id: str
    username: str
    score: float  # -1 (negative) to 1 (positive)
    rationale: str


class SentimentAnalyzer:
    """Batched sentiment analysis via Ollama LLM."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration.

        Args:
            config: Dictionary with 'ollama' and 'sentiment' sections
        """
        ollama_config = config.get('ollama', {})
        sentiment_config = config.get('sentiment', {})

        self.model = ollama_config.get('model', 'qwen3:8b')
        self.temperature = ollama_config.get('temperature', 0)
        self.batch_size = sentiment_config.get('batch_size', 20)

        # Validate batch_size to prevent context window overflow
        if not 1 <= self.batch_size <= 100:
            raise ValueError(
                f"batch_size must be between 1 and 100, got {self.batch_size}"
            )

    @traceable(name="sentiment_analyze_single")
    def analyze_single(
        self,
        comment: str,
        post_title: str,
        post_body: str = ""
    ) -> Dict[str, Any]:
        """Analyze a single comment for sentiment (for evaluation).

        Args:
            comment: The comment text to analyze
            post_title: Title of the Reddit post
            post_body: Body text of the post (optional)

        Returns:
            Dictionary with 'score' (-1 to 1) and 'rationale'
        """
        comments = [{"id": "eval", "author": "user", "body": comment}]
        results = self.analyze_batch(comments, post_title, post_body)

        if results:
            return {
                "score": results[0].score,
                "rationale": results[0].rationale
            }
        return {"score": 0.0, "rationale": "Analysis failed"}

    @traceable(name="sentiment_analyze_batch")
    def analyze_batch(
        self,
        comments: List[Dict[str, Any]],
        post_title: str,
        post_body: str = ""
    ) -> List[SentimentResult]:
        """Analyze a batch of comments for sentiment.

        Args:
            comments: List of comment dicts with 'id', 'author', 'body'
            post_title: Title of the Reddit post
            post_body: Body text of the post (optional)

        Returns:
            List of SentimentResult objects with scores and rationales
        """
        prompt = self._build_prompt(comments, post_title, post_body)

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={'temperature': self.temperature}
        )

        return self._parse_response(response['response'], comments)

    def analyze_all(
        self,
        comments: List[Dict[str, Any]],
        post_title: str,
        post_body: str = ""
    ) -> List[SentimentResult]:
        """Analyze all comments in batches.

        Args:
            comments: List of all comment dicts
            post_title: Title of the Reddit post
            post_body: Body text of the post (optional)

        Returns:
            List of SentimentResult objects for all comments
        """
        all_results = []

        # Split into batches
        for i in range(0, len(comments), self.batch_size):
            batch = comments[i:i + self.batch_size]
            results = self.analyze_batch(batch, post_title, post_body)
            all_results.extend(results)

        return all_results

    def _build_prompt(
        self,
        comments: List[Dict[str, Any]],
        post_title: str,
        post_body: str
    ) -> str:
        """Build the prompt for sentiment analysis.

        Args:
            comments: Batch of comments to analyze
            post_title: Title of the post
            post_body: Body of the post

        Returns:
            Formatted prompt string
        """
        # Format comments for the prompt
        comment_lines = []
        for i, comment in enumerate(comments, 1):
            comment_lines.append(
                f'[{comment["id"]}] u/{comment["author"]}: "{comment["body"]}"'
            )
        comments_text = "\n".join(comment_lines)

        # Truncate post body if too long
        body_preview = post_body[:500] if post_body else "(no body text)"

        prompt = f"""You are analyzing Reddit comments for sentiment toward the original post.

POST TITLE: {post_title}
POST BODY: {body_preview}

COMMENTS TO ANALYZE:
{comments_text}

For each comment, determine the sentiment toward the post/idea on a scale from -1 (negative/dismissive) to 1 (positive/enthusiastic).

Return a JSON array with:
- id: the comment ID (e.g., "c1")
- score: sentiment from -1 to 1
- rationale: brief explanation (10 words max)

Respond ONLY with valid JSON array. Example:
[
  {{"id": "c1", "score": 0.8, "rationale": "Enthusiastic endorsement"}},
  {{"id": "c2", "score": -0.4, "rationale": "Dismissive comparison"}}
]"""

        return prompt

    def _parse_response(
        self,
        response_text: str,
        comments: List[Dict[str, Any]]
    ) -> List[SentimentResult]:
        """Parse LLM response into SentimentResult objects.

        Args:
            response_text: Raw text response from Ollama
            comments: Original comments (for username lookup)

        Returns:
            List of SentimentResult objects
        """
        # Build username lookup
        id_to_author = {c['id']: c['author'] for c in comments}

        # Clean response - remove <think> tags from reasoning models (e.g., Qwen)
        cleaned = response_text.strip()
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()

        # Remove markdown code blocks if present
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json)?\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned)

        # Parse JSON
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response_text}")

        # Convert to SentimentResult objects
        results = []
        for item in data:
            result = SentimentResult(
                comment_id=item['id'],
                username=id_to_author.get(item['id'], 'unknown'),
                score=float(item['score']),
                rationale=item.get('rationale', '')
            )
            results.append(result)

        return results
