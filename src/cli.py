import re

import click
import yaml
from pathlib import Path
from rich.console import Console
from rich.progress import Progress
from dotenv import load_dotenv

from .reddit_client import RedditClient
from .persona_generator import PersonaGenerator
from .vector_store import VectorStore
from .markdown_parser import parse_comments_file, parse_persona_file

console = Console()
load_dotenv()


@click.group()
def cli():
    """Reddit User Research CLI"""
    pass

@cli.command()
@click.argument('userfile', type=click.Path(exists=True, path_type=Path))
@click.option('--config', '-c', default='config/settings.yaml', 
              type=click.Path(path_type=Path), help='Configuration file path')
@click.option('--output-dir', '-o', default='data/output', 
              type=click.Path(path_type=Path), help='Output directory')
@click.option('--skip-existing', is_flag=True, default=True,
              help='Skip users that already have output files')
def fetch(userfile, config, output_dir, skip_existing):
    """Fetch Reddit comments for users in USERFILE and save to markdown."""
    
    # Load configuration
    with open(config, 'r') as f:
        settings = yaml.safe_load(f)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read usernames
    usernames = parse_usernames(userfile)
    
    console.print(f"[bold blue]Fetching comments for {len(usernames)} users...[/bold blue]")
    
    # Filter out users that already have files
    if skip_existing:
        to_process = []
        skipped = []
        for username in usernames:
            markdown_path = output_dir / f"{username}.md"
            if markdown_path.exists():
                skipped.append(username)
            else:
                to_process.append(username)
        
        if skipped:
            console.print(f"[yellow]Skipping {len(skipped)} users with existing files[/yellow]")
        
        usernames = to_process
    
    if not usernames:
        console.print("[yellow]No users to process![/yellow]")
        return
    
    console.print(f"[green]Processing {len(usernames)} users...[/green]")
    
    # Initialize Reddit client
    reddit_client = RedditClient(settings['reddit'])
    
    with Progress() as progress:
        task = progress.add_task("Fetching comments...", total=len(usernames))
        
        for username in usernames:
            try:
                console.print(f"Processing u/{username}")
                
                # Fetch comments
                comments = reddit_client.get_user_comments(username)
                
                # Save to markdown
                markdown_path = output_dir / f"{username}.md"
                reddit_client.save_comments_to_markdown(comments, username, markdown_path)
                
                console.print(f"[green]✓[/green] u/{username}: {len(comments)} comments saved")
                
            except Exception as e:
                console.print(f"[red]✗[/red] u/{username}: {e}")
            
            progress.advance(task)
    
    console.print(f"[bold green]Comment fetching complete! Results in {output_dir}[/bold green]")


@cli.command()
@click.option('--config', '-c', default='config/settings.yaml', 
              type=click.Path(path_type=Path), help='Configuration file path')
@click.option('--input-dir', '-i', default='data/output', 
              type=click.Path(path_type=Path), help='Directory with comment markdown files')
@click.option('--skip-existing', is_flag=True, default=True,
              help='Skip users that already have persona files')
def personas(config, input_dir, skip_existing):
    """Generate personas from existing comment markdown files."""
    
    # Load configuration
    with open(config, 'r') as f:
        settings = yaml.safe_load(f)
    
    # Find all comment markdown files
    comment_files = list(input_dir.glob("*.md"))
    comment_files = [f for f in comment_files if not f.name.lower().endswith('_persona.md') and not f.name.endswith('_test.md')]
    
    if not comment_files:
        console.print(f"[red]No comment files found in {input_dir}[/red]")
        return
    
    console.print(f"[bold blue]Found {len(comment_files)} comment files[/bold blue]")
    
    # Filter out users that already have personas
    to_analyze = []
    skipped = []
    for comment_file in comment_files:
        username = comment_file.stem
        persona_file = input_dir / f"{username}_persona.md"
        if skip_existing and persona_file.exists():
            skipped.append(username)
        else:
            to_analyze.append((username, comment_file))
    
    if skipped:
        console.print(f"[yellow]Skipping {len(skipped)} users with existing personas[/yellow]")
    
    if not to_analyze:
        console.print("[yellow]All users already have personas![/yellow]")
        return
    
    console.print(f"[green]Generating personas for {len(to_analyze)} users...[/green]")
    
    # Initialize persona generator
    persona_generator = PersonaGenerator(settings)
    
    with Progress() as progress:
        persona_task = progress.add_task("Analyzing personas...", total=len(to_analyze))
        
        for username, comment_file in to_analyze:
            try:
                console.print(f"Analyzing u/{username}...")
                
                # Generate persona
                persona = persona_generator.generate_persona(comment_file)
                
                # Save persona
                persona_path = input_dir / f"{username}_persona.md"
                with open(persona_path, 'w', encoding='utf-8') as f:
                    f.write(persona)
                
                console.print(f"[green]✓[/green] u/{username} persona generated")
                
            except Exception as e:
                console.print(f"[red]✗[/red] u/{username} persona failed: {e}")
            
            progress.advance(persona_task)
    
    console.print(f"[bold green]Persona generation complete![/bold green]")


@cli.command()
@click.option('--config', '-c', default='config/settings.yaml',
              type=click.Path(path_type=Path), help='Configuration file path')
@click.option('--input-dir', '-i', default='data/output',
              type=click.Path(path_type=Path), help='Directory with markdown files')
@click.option('--collection', type=click.Choice(['comments', 'personas', 'all']),
              default='all', help='Which collection to embed')
@click.option('--user', '-u', default=None, help='Embed only specific user')
@click.option('--skip-existing/--no-skip-existing', default=True,
              help='Skip files already embedded (based on Qdrant count)')
def embed(config, input_dir, collection, user, skip_existing):
    """Embed markdown files into Qdrant vector database."""

    # Load configuration
    with open(config, 'r') as f:
        settings = yaml.safe_load(f)

    # Initialize vector store
    store = VectorStore(settings)
    store.initialize_collections()

    console.print("[bold blue]Embedding data into Qdrant...[/bold blue]")

    # Determine which files to process
    if user:
        comment_files = [input_dir / f"{user}.md"] if (input_dir / f"{user}.md").exists() else []
        persona_files = [input_dir / f"{user}_Persona.md"] if (input_dir / f"{user}_Persona.md").exists() else []
    else:
        comment_files = [f for f in input_dir.glob("*.md")
                        if not f.name.lower().endswith('_persona.md')]
        persona_files = list(input_dir.glob("*_Persona.md"))

    # Embed comments
    if collection in ['comments', 'all'] and comment_files:
        # Filter out users already embedded if skip_existing is True
        if skip_existing:
            files_to_embed = []
            skipped_count = 0
            for f in comment_files:
                username = f.stem
                if store.user_has_comments(username):
                    skipped_count += 1
                else:
                    files_to_embed.append(f)
            if skipped_count > 0:
                console.print(f"[yellow]Skipping {skipped_count} users already embedded[/yellow]")
            comment_files = files_to_embed

        if comment_files:
            console.print(f"[green]Embedding {len(comment_files)} comment files...[/green]")

            with Progress() as progress:
                task = progress.add_task("Embedding comments...", total=len(comment_files))

                total_comments = 0
                for comment_file in comment_files:
                    try:
                        username = comment_file.stem
                        comments = parse_comments_file(comment_file)

                        for comment in comments:
                            store.store_comment(comment, username=username)
                            total_comments += 1

                        console.print(f"[green]✓[/green] {username}: {len(comments)} comments")

                    except Exception as e:
                        console.print(f"[red]✗[/red] {comment_file.name}: {e}")

                    progress.advance(task)

            console.print(f"[bold green]Embedded {total_comments} comments[/bold green]")

    # Embed personas
    if collection in ['personas', 'all'] and persona_files:
        # Filter out users already embedded if skip_existing is True
        if skip_existing:
            files_to_embed = []
            skipped_count = 0
            for f in persona_files:
                # Extract username from filename (e.g., "Username_Persona.md" -> "Username")
                username = f.stem.replace('_Persona', '')
                if store.user_has_persona(username):
                    skipped_count += 1
                else:
                    files_to_embed.append(f)
            if skipped_count > 0:
                console.print(f"[yellow]Skipping {skipped_count} personas already embedded[/yellow]")
            persona_files = files_to_embed

        if persona_files:
            console.print(f"[green]Embedding {len(persona_files)} persona files...[/green]")

            with Progress() as progress:
                task = progress.add_task("Embedding personas...", total=len(persona_files))

                for persona_file in persona_files:
                    try:
                        persona = parse_persona_file(persona_file)

                        # Count comments from corresponding comment file
                        comment_file = input_dir / f"{persona['username']}.md"
                        comment_count = 0
                        if comment_file.exists():
                            comments = parse_comments_file(comment_file)
                            comment_count = len(comments)

                        store.store_persona(
                            username=persona['username'],
                            persona_text=persona['persona_text'],
                            archetype=persona['archetype'],
                            top_subreddits=persona['top_subreddits'],
                            comment_count=comment_count
                        )

                        console.print(f"[green]✓[/green] {persona['username']} persona")

                    except Exception as e:
                        console.print(f"[red]✗[/red] {persona_file.name}: {e}")

                    progress.advance(task)

            console.print(f"[bold green]Embedded {len(persona_files)} personas[/bold green]")

    console.print("[bold green]Embedding complete![/bold green]")


@cli.command()
@click.argument('query')
@click.option('--config', '-c', default='config/settings.yaml',
              type=click.Path(path_type=Path), help='Configuration file path')
@click.option('--collection', type=click.Choice(['comments', 'personas']),
              default='comments', help='Which collection to search')
@click.option('--limit', '-n', default=10, help='Number of results')
def search(query, config, collection, limit):
    """Semantic search across embedded data."""

    # Load configuration
    with open(config, 'r') as f:
        settings = yaml.safe_load(f)

    # Initialize vector store
    store = VectorStore(settings)

    console.print(f"[bold blue]Searching {collection} for:[/bold blue] {query}")

    results = store.search_similar(query, collection=collection, limit=limit)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"\n[bold]Found {len(results)} results:[/bold]\n")

    for i, result in enumerate(results, 1):
        similarity = result.get('similarity', 0)

        if collection == 'comments':
            text = result.get('text', '')[:200]
            username = result.get('username', 'Unknown')
            subreddit = result.get('subreddit', '')
            reddit_score = result.get('score', 0)

            console.print(f"[bold cyan]{i}.[/bold cyan] [dim](similarity: {similarity:.3f})[/dim]")
            console.print(f"   [bold]u/{username}[/bold] in r/{subreddit} [dim](+{reddit_score})[/dim]")
            console.print(f"   {text}...")
            console.print()
        else:
            username = result.get('username', 'Unknown')
            archetype = result.get('archetype', '')

            console.print(f"[bold cyan]{i}.[/bold cyan] [dim](similarity: {similarity:.3f})[/dim]")
            console.print(f"   [bold]u/{username}[/bold] - {archetype}")
            console.print()


@cli.command()
@click.argument('question')
@click.option('--config', '-c', default='config/settings.yaml',
              type=click.Path(path_type=Path), help='Configuration file path')
@click.option('--limit', '-n', default=10, help='Number of context items to retrieve')
def ask(question, config, limit):
    """Ask a question about your audience using RAG."""
    import ollama as ollama_client

    # Load configuration
    with open(config, 'r') as f:
        settings = yaml.safe_load(f)

    # Initialize vector store
    try:
        store = VectorStore(settings)
    except Exception as e:
        console.print(f"[red]Failed to connect to Qdrant:[/red] {e}")
        console.print("[dim]Make sure Qdrant is running: ./scripts/setup_qdrant.sh start[/dim]")
        return

    console.print(f"[bold blue]Question:[/bold blue] {question}")
    console.print("[dim]Retrieving relevant context...[/dim]")

    # Retrieve context from both collections
    try:
        comment_results = store.search_similar(question, collection="comments", limit=limit)
        persona_results = store.search_similar(question, collection="personas", limit=5)
    except Exception as e:
        console.print(f"[red]Failed to search Qdrant:[/red] {e}")
        console.print("[dim]Have you run 'python main.py embed' first?[/dim]")
        return

    if not comment_results and not persona_results:
        console.print("[yellow]No relevant context found in the database.[/yellow]")
        console.print("[dim]Make sure you've embedded data with 'python main.py embed'[/dim]")
        return

    # Build context
    context_parts = []

    if persona_results:
        context_parts.append("## Relevant User Personas\n")
        for r in persona_results:
            context_parts.append(f"**u/{r.get('username', 'Unknown')}** ({r.get('archetype', '')}):\n")
            # Include a snippet of the persona
            persona_text = r.get('persona_text', '')[:500]
            context_parts.append(f"{persona_text}\n\n")

    if comment_results:
        context_parts.append("## Relevant Comments\n")
        for r in comment_results:
            username = r.get('username', 'Unknown')
            subreddit = r.get('subreddit', '')
            text = r.get('text', '')
            score = r.get('score', 0)
            context_parts.append(f"**u/{username}** in r/{subreddit} (score: {score}):\n")
            context_parts.append(f"> {text}\n\n")

    context = "\n".join(context_parts)

    # Build RAG prompt
    prompt = f"""You are analyzing Reddit user data to answer questions about an audience.

Use the following context from real Reddit comments and user personas to answer the question.
Cite specific examples from the comments when relevant. Be specific and grounded in the data.

{context}

---

Question: {question}

Answer:"""

    console.print("[dim]Generating answer...[/dim]\n")

    # Call Ollama
    ollama_settings = settings.get('ollama', {})
    model = ollama_settings.get('model', 'qwen3:8b')

    try:
        response = ollama_client.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': ollama_settings.get('temperature', 0.3)
            }
        )
    except Exception as e:
        console.print(f"[red]Failed to connect to Ollama:[/red] {e}")
        console.print("[dim]Make sure Ollama is running: ollama serve[/dim]")
        return

    # Clean response (remove <think> tags if present)
    answer = response.get('response', '')
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

    console.print("[bold green]Answer:[/bold green]")
    console.print(answer)


def parse_usernames(userfile: Path) -> list[str]:
    """
    Parse usernames from file, handling various formats.
    
    Expected formats:
    - Plain usernames (one per line)
    - Numbered list (1→username)
    - Mixed with empty lines
    """
    usernames = []
    
    with open(userfile, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Handle numbered format: "1→username"
            if '→' in line:
                parts = line.split('→', 1)
                if len(parts) == 2:
                    username = parts[1].strip()
                    if username:  # Only add non-empty usernames
                        usernames.append(username)
            else:
                # Handle plain username format
                if line:
                    usernames.append(line)
    
    return usernames


def main():
    cli()


if __name__ == "__main__":
    main()