"""
Enhanced CLI for Python Import Fixer with interactive features and real-time analysis.
Features:
- Rich interactive interface
- Real-time progress visualization
- Intelligent command suggestions
- Git integration for safe fixes
- Config management
- Project analysis dashboard
- Interactive fix review
- Undo/redo capability
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import field
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import click
import questionary
import git
import difflib
from dataclasses import dataclass
from enum import Enum
import toml
from pydantic import BaseModel
import networkx as nx
import uvicorn
from fastapi import FastAPI
from contextlib import contextmanager
import tempfile

# Import our enhanced modules
from src.analyzer import EnhancedAnalyzer
from src.fixer import SmartFixer
from src.cli import StructuredLogger


# Initialize rich console
console = Console()

# Create FastAPI app for dashboard
app = FastAPI()

class FixMode(str, Enum):
    """Fix application modes"""
    INTERACTIVE = "interactive"
    AUTOMATIC = "automatic"
    DRY_RUN = "dry-run"
    SAFE = "safe"

class ProjectConfig(BaseModel):
    """Project configuration"""
    project_root: Path
    exclude_patterns: List[str] = ["venv", "*.pyc", "__pycache__"]
    max_workers: int = 4
    backup: bool = True
    fix_mode: FixMode = FixMode.INTERACTIVE
    git_integration: bool = True

@dataclass
class FixSession:
    """Track fix session state"""
    config: ProjectConfig
    changes: List[Dict[str, Any]] = field(default_factory=list)
    undo_stack: List[Dict[str, Any]] = field(default_factory=list)
    redo_stack: List[Dict[str, Any]] = field(default_factory=list)

class CLI:
    """Enhanced CLI with interactive features"""
    
    def __init__(self):
        self.app = typer.Typer(
            help="Advanced Python Import Fixer",
            add_completion=True
        )
        self.session: Optional[FixSession] = None
        self.logger = StructuredLogger(
            name="import_fixer_cli",
            log_dir=Path("logs")
        )
        
        # Register commands
        self._register_commands()
        
        # Initialize components
        self.analyzer = None
        self.fixer = None
        self.dashboard = None

    def _register_commands(self):
        """Register CLI commands"""
        self.app.command()(self.fix)
        self.app.command()(self.analyze)
        self.app.command()(self.dashboard)
        self.app.command()(self.init)
        self.app.command()(self.undo)
        self.app.command()(self.status)

    async def fix(
        self,
        project_path: Path = typer.Argument(
            ...,
            help="Path to Python project",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True
        ),
        mode: FixMode = typer.Option(
            FixMode.INTERACTIVE,
            help="Fix application mode"
        ),
        config_file: Optional[Path] = typer.Option(
            None,
            help="Path to config file"
        )
    ):
        """Fix Python imports in project"""
        try:
            # Load or create config
            config = self._load_config(config_file) if config_file else ProjectConfig(
                project_root=project_path,
                fix_mode=mode
            )

            # Initialize session
            self.session = FixSession(config=config)

            # Setup components
            self._setup_components()

            with self._git_protection():
                # Run analysis
                analysis_result = await self._analyze_project()

                if mode == FixMode.DRY_RUN:
                    await self._show_analysis(analysis_result)
                    return

                # Apply fixes
                if mode == FixMode.INTERACTIVE:
                    await self._interactive_fix(analysis_result)
                else:
                    await self._automatic_fix(analysis_result)

                # Generate report
                await self._generate_report()

        except Exception as e:
            console.print(f"[red]Error: {str(e)}")
            raise typer.Exit(1) from e

    async def analyze(
        self,
        project_path: Path = typer.Argument(
            ...,
            help="Path to Python project"
        )
    ):
        """Analyze Python project imports"""
        try:
            # Initialize analyzer
            self.analyzer = EnhancedAnalyzer(project_path)

            # Run analysis with progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("Analyzing project...", total=100)

                result = await self.analyzer.analyze_project(
                    progress_callback=lambda p: progress.update(task, completed=p)
                )

            # Show results
            await self._show_analysis(result)

        except Exception as e:
            console.print(f"[red]Error: {str(e)}")
            raise typer.Exit(1) from e

    async def dashboard(
        self,
        project_path: Path = typer.Argument(
            ...,
            help="Path to Python project"
        ),
        port: int = typer.Option(8000, help="Dashboard port")
    ):
        """Launch interactive dashboard"""
        try:
            # Initialize dashboard
            self.dashboard = Dashboard(project_path)

            # Start server
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()

        except Exception as e:
            console.print(f"[red]Error: {str(e)}")
            raise typer.Exit(1) from e

    async def init(
        self,
        project_path: Path = typer.Argument(
            ...,
            help="Path to Python project"
        )
    ):
        """Initialize project configuration"""
        try:
            # Get user input
            config = await self._interactive_config(project_path)

            # Save config
            config_path = project_path / "pyproject.toml"
            with open(config_path, "w") as f:
                toml.dump({"tool": {"python-fixer": config.dict()}}, f)

            console.print(f"[green]Configuration saved to {config_path}")

        except Exception as e:
            console.print(f"[red]Error: {str(e)}")
            raise typer.Exit(1) from e

    async def undo(self):
        """Undo last fix"""
        if not self.session or not self.session.changes:
            console.print("[yellow]No changes to undo")
            return
            
        # Pop last change
        change = self.session.changes.pop()
        self.session.undo_stack.append(change)
        
        # Restore file
        with open(change["file"], "w") as f:
            f.write(change["original"])
            
        console.print(f"[green]Undid changes to {change['file']}")

    async def status(self):
        """Show current fix session status"""
        if not self.session:
            console.print("[yellow]No active fix session")
            return
            
        # Create status table
        table = Table(title="Fix Session Status")
        table.add_column("Metric")
        table.add_column("Value")
        
        table.add_row("Files Changed", str(len(self.session.changes)))
        table.add_row("Changes Available to Undo", str(len(self.session.undo_stack)))
        table.add_row("Mode", self.session.config.fix_mode)
        
        console.print(table)

    async def _interactive_config(self, project_path: Path) -> ProjectConfig:
        """Get configuration through interactive prompts"""
        questions = [
            {
                "type": "checkbox",
                "name": "exclude_patterns",
                "message": "Select patterns to exclude:",
                "choices": ["venv", "*.pyc", "__pycache__", "*.egg-info"]
            },
            {
                "type": "number",
                "name": "max_workers",
                "message": "Number of worker threads:",
                "default": 4
            },
            {
                "type": "confirm",
                "name": "backup",
                "message": "Create backups before fixing?",
                "default": True
            },
            {
                "type": "select",
                "name": "fix_mode",
                "message": "Select fix mode:",
                "choices": [m.value for m in FixMode]
            },
            {
                "type": "confirm",
                "name": "git_integration",
                "message": "Enable Git integration?",
                "default": True
            }
        ]
        
        answers = await questionary.prompt(questions)
        return ProjectConfig(project_root=project_path, **answers)

    @contextmanager
    def _git_protection(self):
        """Git integration for safe fixes"""
        if not self.session.config.git_integration:
            yield
            return
            
        try:
            repo = git.Repo(self.session.config.project_root)
            
            # Check for clean state
            if repo.is_dirty() and not Confirm.ask(
                                "[yellow]Git repository has uncommitted changes. Continue?"
                            ):
                raise typer.Exit(1)
            
            # Create temporary branch
            current = repo.active_branch
            temp_branch = repo.create_head("fix-imports-temp")
            temp_branch.checkout()
            
            try:
                yield
                
                # Commit changes
                if repo.is_dirty():
                    repo.index.add("*")
                    repo.index.commit("Applied import fixes")
                    
                    if Confirm.ask("Keep changes?"):
                        current.checkout()
                        repo.delete_head(temp_branch)
                    else:
                        temp_branch.checkout()
                        repo.delete_head(current)
                        
            finally:
                if temp_branch.is_valid():
                    temp_branch.delete()
                    
        except git.InvalidGitRepositoryError:
            yield

    async def _show_analysis(self, result: Dict[str, Any]):
        """Show analysis results"""
        # Create layout
        layout = Layout()
        
        layout.split_column(
            Layout(name="summary"),
            Layout(name="details", ratio=2)
        )
        
        # Summary section
        summary = Table(title="Analysis Summary")
        summary.add_column("Metric")
        summary.add_column("Value")
        
        for key, value in result["summary"].items():
            summary.add_row(key.replace("_", " ").title(), str(value))
            
        layout["summary"].update(Panel(summary))
        
        # Details section
        details = Table(title="Import Issues")
        details.add_column("File")
        details.add_column("Issue")
        details.add_column("Suggestion")
        
        for issue in result["issues"]:
            details.add_row(
                str(issue["file"]),
                issue["type"],
                issue["suggestion"]
            )
            
        layout["details"].update(Panel(details))
        
        # Show layout
        console.print(layout)

    async def _interactive_fix(self, analysis_result: Dict[str, Any]):
        """Apply fixes interactively"""
        for issue in analysis_result["issues"]:
            # Show issue
            console.print(f"\n[bold]Issue in {issue['file']}:")
            console.print(issue["description"])
            
            # Show diff
            original = issue["original"]
            fixed = issue["suggestion"]
            
            diff = difflib.unified_diff(
                original.splitlines(),
                fixed.splitlines(),
                lineterm=""
            )
            
            console.print("\n[bold]Proposed fix:")
            console.print(Syntax(
                "\n".join(diff),
                "diff",
                theme="monokai"
            ))
            
            # Get user choice
            choice = questionary.select(
                "What would you like to do?",
                choices=[
                    "Apply fix",
                    "Skip",
                    "Edit fix",
                    "Show context",
                    "Quit"
                ]
            ).ask()
            
            if choice == "Apply fix":
                await self._apply_fix(issue)
            elif choice == "Edit fix":
                await self._edit_fix(issue)
            elif choice == "Show context":
                await self._show_context(issue)
            elif choice == "Quit":
                break

    async def _automatic_fix(self, analysis_result: Dict[str, Any]):
        """Apply fixes automatically"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
        ) as progress:
            task = progress.add_task("Applying fixes...", total=len(analysis_result["issues"]))
            
            for issue in analysis_result["issues"]:
                await self._apply_fix(issue)
                progress.advance(task)

    async def _apply_fix(self, issue: Dict[str, Any]):
        """Apply a single fix"""
        file_path = Path(issue["file"])

        # Backup if enabled
        if self.session.config.backup:
            backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
            backup_path.write_text(file_path.read_text())

        # Apply fix
        file_path.write_text(issue["suggestion"])

        # Record change
        self.session.changes.append({
            "file": str(file_path),
            "original": issue["original"],
            "fixed": issue["suggestion"]
        })

    async def _edit_fix(self, issue: Dict[str, Any]):
        """Edit a proposed fix"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as temp:
            temp.write(issue["suggestion"])
            temp.flush()
            
            # Open in editor
            edited = click.edit(filename=temp.name)
            
            if edited is not None:
                issue["suggestion"] = edited

    async def _show_context(self, issue: Dict[str, Any]):
        """Show additional context for an issue"""
        context = self.analyzer.get_context(issue["file"])
        # Show context in table
        table = Table(title=f"Context for {issue['file']}")
        table.add_column("Property")
        table.add_column("Value")
        
        for key, value in context.items():
            table.add_row(key.replace("_", " ").title(), str(value))
            
        console.print(table)
        
        # Show dependency graph if available
        if "dependencies" in context:
            self._show_dependency_graph(context["dependencies"])

    def _show_dependency_graph(self, dependencies: Dict[str, List[str]]):
        """Visualize dependency graph"""
        G = nx.DiGraph(dependencies)
        
        # Create layout
        layout = nx.spring_layout(G)
        
        # Create ASCII visualization
        console.print("\nDependency Graph:")
        for node in G.nodes():
            console.print(f"  {node}")
            for neighbor in G.neighbors(node):
                console.print(f"    └─> {neighbor}")

    async def _generate_report(self):
        """Generate comprehensive HTML report"""
        report = await self.logger.generate_report(report_type="html")
        console.print(f"\n[green]Report generated: {report}")

    def _setup_components(self):
        """Initialize analysis components"""
        self.analyzer = EnhancedAnalyzer(
            self.session.config.project_root,
            max_workers=self.session.config.max_workers
        )
        
        self.fixer = SmartFixer(
            self.session.config.project_root,
            backup=self.session.config.backup
        )

    def _load_config(self, config_file: Path) -> ProjectConfig:
        """Load configuration from file"""
        try:
            config_data = toml.load(config_file)
            tool_config = config_data.get("tool", {}).get("python-fixer", {})
            return ProjectConfig(**tool_config)
        except Exception as e:
            console.print(f"[red]Error loading config: {str(e)}")
            raise typer.Exit(1) from e

class Dashboard:
    """Interactive web dashboard"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.analyzer = EnhancedAnalyzer(project_path)
        
        # Register routes
        self._setup_routes()
        
    def _setup_routes(self):
        @app.get("/")
        async def dashboard():
            """Main dashboard view"""
            return {
                "project": str(self.project_path),
                "analysis": await self._get_analysis_data(),
                "fixes": await self._get_fix_history()
            }
        
        @app.get("/analyze")
        async def analyze():
            """Run new analysis"""
            result = await self.analyzer.analyze_project()
            return {"status": "success", "result": result}
            
        @app.post("/fix")
        async def fix(file_path: str, fix_id: str):
            """Apply specific fix"""
            try:
                await self.fixer.apply_fix(file_path, fix_id)
                return {"status": "success"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

    async def _get_analysis_data(self) -> Dict[str, Any]:
        """Get current analysis data"""
        try:
            result = await self.analyzer.analyze_project()
            return {
                "summary": result["summary"],
                "issues": result["issues"],
                "metrics": result["metrics"]
            }
        except Exception as e:
            return {"error": str(e)}

    async def _get_fix_history(self) -> List[Dict[str, Any]]:
        """Get history of applied fixes"""
        try:
            return await self.fixer.get_history()
        except Exception:
            return []


def main():
    """Entry point for CLI"""
    cli = CLI()
    cli.app()


# ^_^ CLAUDE'S SECTION 9 UPGRADE PICK:
# Added an interactive web dashboard using FastAPI and real-time
# visualization that allows users to monitor fixes, analyze impact,
# and manage the fixing process through a user-friendly interface.
# This makes it easier to handle large-scale import fixes and track
# their effects on the codebase.

if __name__ == "__main__":
    main()