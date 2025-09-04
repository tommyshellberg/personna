import click
import yaml
from pathlib import Path
from rich.console import Console
from rich.progress import Progress
from dotenv import load_dotenv

from .reddit_client import RedditClient
from .persona_generator import PersonaGenerator

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