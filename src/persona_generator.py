import ollama
import re
from typing import Dict, Any
from pathlib import Path
from datetime import datetime


class PersonaGenerator:
    """Generates user personas using local LLM analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_config = config['ollama']
        self.client = ollama.Client(host=self.ollama_config['base_url'])
    
    def _clean_llm_response(self, response: str) -> str:
        """Remove thinking tags and clean up LLM response."""
        # Remove <think>...</think> blocks
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()
    
    def generate_persona(self, markdown_path: Path) -> str:
        """Generate a user persona from their Reddit comments."""
        
        # Read the user's comments
        with open(markdown_path, 'r', encoding='utf-8') as f:
            comments_content = f.read()
        
        # Extract username from filename
        username = markdown_path.stem
        
        # Create persona generation prompt
        prompt = self._create_persona_prompt(username, comments_content)
        
        try:
            response = self.client.generate(
                model=self.ollama_config['model'],
                prompt=prompt,
                options={
                    'temperature': self.ollama_config.get('temperature', 0.3),
                    'top_p': 0.9,
                    'num_ctx': 32768  # Use ~32K context to handle large comment files
                }
            )
            
            # Clean and format the response
            raw_response = response['response']
            cleaned_response = self._clean_llm_response(raw_response)
            
            # Format as markdown
            persona_md = f"# User Persona: u/{username}\n\n"
            persona_md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            persona_md += cleaned_response
            
            return persona_md
            
        except Exception as e:
            raise Exception(f"Failed to generate persona for {username}: {e}")
    
    def _create_persona_prompt(self, username: str, comments_content: str) -> str:
        """Create a structured prompt for persona generation."""
        
        archetypes = [
            "The Innocent", "The Everyman", "The Hero", "The Caregiver", 
            "The Explorer", "The Rebel", "The Lover", "The Creator", 
            "The Jester", "The Sage", "The Magician", "The Ruler"
        ]
        
        prompt = f"""
Analyze the Reddit comments below for user u/{username} and create a comprehensive user persona.

REDDIT COMMENTS DATA:
{comments_content}

Please provide a structured analysis in the following format:

## User Persona Summary
Write 2-3 sentences describing this user's overall personality and online presence.

## Demographics & Background
- **Likely Age Range:** [age range with reasoning]
- **Possible Occupation/Field:** [based on language, interests, time patterns]
- **Technical Level:** [beginner/intermediate/advanced in tech topics]

## Communication Style
- **Tone:** [formal/casual/humorous/technical/etc.]
- **Language Patterns:** [specific phrases, technical jargon, emotional expressions]
- **Engagement Style:** [how they interact - helpful, argumentative, supportive, etc.]

## Interests & Topics
List the main topics this user discusses and seems passionate about.

## Jungian Archetype
Choose the most fitting archetype from: {', '.join(archetypes)}
Explain why this archetype fits and what it means for engagement.

## Subreddit Activity Analysis
- **Most Active Communities:** [list top subreddits with engagement patterns]
- **Community Role:** [lurker/contributor/expert/newcomer in each community]

## Engagement Recommendations
- **Content Types:** [what kind of posts would appeal - memes, tutorials, discussions, etc.]
- **Communication Approach:** [how to talk to them - technical depth, humor style, etc.]
- **Best Subreddits to Reach Similar Users:** [where to find people like them]

Base your analysis only on the provided comments. Be specific and actionable in recommendations.
"""
        return prompt