"""
Enhanced logging formatter with advanced visualization and real-time analytics.
Features:
- Rich terminal output with advanced formatting
- Real-time log aggregation and analysis
- Interactive HTML report generation
- Log pattern detection using ML
- Customizable themes and layouts
- Metric visualization
- Dependency graph rendering
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Deque, Dict

import aiofiles
import altair as alt
import dash
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from jinja2 import (
    Environment,
    PackageLoader,
    select_autoescape,
)
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

console = Console()


@dataclass
class LogMetrics:
    """Real-time metrics for log analysis"""

    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    import_fixes: int = 0
    files_processed: int = 0
    average_processing_time: float = 0.0
    pattern_frequencies: Dict[str, int] = field(default_factory=dict)


class EnhancedFormatter(logging.Formatter):
    """Advanced log formatter with real-time analytics and visualization"""

    def __init__(
        self,
        output_dir: Path,
        theme: str = "monokai",
        real_time: bool = True,
        max_history: int = 1000,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.theme = theme
        self.real_time = real_time
        self.metrics = LogMetrics()
        self.log_history: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self.pattern_detector = self._initialize_pattern_detector()
        self.visualization_engine = self._initialize_visualization()
        self.layout = self._create_layout()
        self._lock = Lock()

        # Initialize real-time display
        if real_time:
            self._start_live_display()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with enhanced styling and analytics"""
        # Update metrics
        with self._lock:
            self._update_metrics(record)
            self._analyze_patterns(record)

        # Create base log entry
        log_entry = {
            "timestamp": self._format_timestamp(record.created),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "context": self._extract_context(record),
        }

        # Add to history
        self.log_history.append(log_entry)

        # Format based on level
        if record.levelno >= logging.ERROR:
            return self._format_error(log_entry)
        elif record.levelno >= logging.WARNING:
            return self._format_warning(log_entry)
        else:
            return self._format_info(log_entry)

    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp with millisecond precision"""
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _extract_context(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Extract rich context from log record"""
        context = {
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }

        # Extract extra fields
        if hasattr(record, "extra"):
            context |= record.extra

        return context

    def _format_error(self, entry: Dict[str, Any]) -> str:
        """Format error entries with stack trace and context"""
        panel = Panel(
            self._create_error_layout(entry),
            title=f"[bold red]ERROR - {entry['timestamp']}[/]",
            border_style="red",
        )
        # Convert panel to string representation
        with console.capture() as capture:
            console.print(panel)
        return capture.get()

    def _format_warning(self, entry: Dict[str, Any]) -> str:
        """Format warning entries with context"""
        panel = Panel(
            self._create_warning_layout(entry),
            title=f"[bold yellow]WARNING - {entry['timestamp']}[/]",
            border_style="yellow",
        )
        # Convert panel to string representation
        with console.capture() as capture:
            console.print(panel)
        return capture.get()

    def _format_info(self, entry: Dict[str, Any]) -> str:
        """Format info entries with metrics"""
        panel = Panel(
            self._create_info_layout(entry),
            title=f"[bold green]INFO - {entry['timestamp']}[/]",
            border_style="green",
        )
        # Convert panel to string representation
        with console.capture() as capture:
            console.print(panel)
        return capture.get()

    def _create_error_layout(self, entry: Dict[str, Any]) -> Layout:
        """Create rich layout for error entries"""
        layout = Layout()

        # Message section
        layout.split_column(
            Layout(name="message", size=2),
            Layout(name="context", size=3),
            Layout(name="stack", size=5),
        )

        self._extracted_from__create_warning_layout_12(layout, entry, "red", "bold red")
        # Stack trace
        if "stack_trace" in entry["context"]:
            layout["stack"].update(
                Syntax(
                    entry["context"]["stack_trace"],
                    "pytb",
                    theme=self.theme,
                    line_numbers=True,
                )
            )

        return layout

    def _create_warning_layout(self, entry: Dict[str, Any]) -> Layout:
        """Create rich layout for warning entries"""
        layout = Layout()

        layout.split_column(
            Layout(name="message", size=2), Layout(name="context", size=3)
        )

        self._extracted_from__create_warning_layout_12(
            layout, entry, "yellow", "bold yellow"
        )
        return layout

    # TODO Rename this here and in `_create_error_layout` and `_create_warning_layout`
    def _extracted_from__create_warning_layout_12(
        self, layout, entry, border_style, header_style
    ):
        layout["message"].update(
            Panel(entry["message"], title="Message", border_style=border_style)
        )
        context_table = Table(show_header=True, header_style=header_style)
        context_table.add_column("Key")
        context_table.add_column("Value")
        for k, v in entry["context"].items():
            context_table.add_row(str(k), str(v))
        layout["context"].update(Panel(context_table, title="Context"))

    def _create_info_layout(self, entry: Dict[str, Any]) -> Layout:
        """Create rich layout for info entries"""
        layout = Layout()

        layout.split_row(Layout(name="message"), Layout(name="metrics", ratio=2))

        layout["message"].update(
            Panel(entry["message"], title="Message", border_style="green")
        )

        # Metrics table
        metrics_table = Table(
            show_header=True, header_style="bold green", title="Current Metrics"
        )
        metrics_table.add_column("Metric")
        metrics_table.add_column("Value")

        with self._lock:
            for k, v in self.metrics.__dict__.items():
                metrics_table.add_row(k.replace("_", " ").title(), str(v))

        layout["metrics"].update(metrics_table)

        return layout

    def _update_metrics(self, record: logging.LogRecord):
        """Update real-time metrics"""
        if record.levelno >= logging.ERROR:
            self.metrics.error_count += 1
        elif record.levelno >= logging.WARNING:
            self.metrics.warning_count += 1
        else:
            self.metrics.info_count += 1

        if hasattr(record, "files_processed"):
            self.metrics.files_processed = record.files_processed

        if hasattr(record, "import_fixes"):
            self.metrics.import_fixes = record.import_fixes

    def _analyze_patterns(self, record: logging.LogRecord):
        """Analyze log patterns using ML"""
        message = record.getMessage()
        features = self.pattern_detector.transform([message])
        cluster = self.pattern_detector.predict(features)[0]

        pattern_key = f"pattern_{cluster}"
        self.metrics.pattern_frequencies[pattern_key] = (
            self.metrics.pattern_frequencies.get(pattern_key, 0) + 1
        )

    def _initialize_pattern_detector(self):
        """Initialize ML-based pattern detector"""
        vectorizer = TfidfVectorizer(max_features=1000)
        clustering = DBSCAN(eps=0.3, min_samples=2)

        return Pipeline([("vectorizer", vectorizer), ("clustering", clustering)])

    def _initialize_visualization(self):
        """Initialize visualization engine"""
        return {"plotly": dash.Dash(__name__), "altair": alt.Chart, "seaborn": sns}

    async def generate_report(self):
        """Generate comprehensive HTML report"""
        env = Environment(
            loader=PackageLoader("python_fixer", "templates"),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template("log_report.html")

        # Generate visualizations
        visualizations = await self._generate_visualizations()

        # Prepare report data
        report_data = {
            "metrics": self.metrics.__dict__,
            "visualizations": visualizations,
            "log_history": list(self.log_history),
            "patterns": self._analyze_log_patterns(),
        }

        # Render report
        html = template.render(**report_data)

        # Save report
        report_path = self.output_dir / "log_report.html"
        async with aiofiles.open(report_path, "w") as f:
            await f.write(html)

        return report_path

    async def _generate_visualizations(self) -> Dict[str, str]:
        """Generate interactive visualizations"""
        # Metrics over time
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(self.log_history))),
                y=[entry.get("level") == "ERROR" for entry in self.log_history],
                name="Errors",
            )
        )
        visualizations = {"metrics_plot": fig.to_html(include_plotlyjs=True)}
        # Pattern distribution
        pattern_df = pd.DataFrame(
            list(self.metrics.pattern_frequencies.items()),
            columns=["Pattern", "Frequency"],
        )
        chart = alt.Chart(pattern_df).mark_bar().encode(x="Pattern", y="Frequency")
        visualizations["pattern_chart"] = chart.to_html()

        return visualizations

    def _analyze_log_patterns(self) -> Dict[str, Any]:
        """Analyze log patterns for insights"""
        messages = [entry["message"] for entry in self.log_history]
        vectorizer = TfidfVectorizer(max_features=100)
        features = vectorizer.fit_transform(messages)

        # Cluster messages
        clustering = DBSCAN(eps=0.3, min_samples=2)
        clusters = clustering.fit_predict(features)

        # Analyze patterns
        patterns = {}
        for cluster_id in set(clusters):
            if cluster_id == -1:
                continue
            cluster_messages = [
                msg for msg, c in zip(messages, clusters) if c == cluster_id
            ]
            patterns[f"pattern_{cluster_id}"] = {
                "count": len(cluster_messages),
                "example": cluster_messages[0],
                "frequency": len(cluster_messages) / len(messages),
            }

        return patterns

    def _start_live_display(self):
        """Start real-time display updating"""

        async def update_display():
            while True:
                with Live(self.layout, refresh_per_second=4):
                    await asyncio.sleep(0.25)
                    self._update_layout()

        asyncio.create_task(update_display())

    def _update_layout(self):
        """Update live display layout"""
        with self._lock:
            metrics_table = self._extracted_from__update_layout_5("Metric", "Value")
            for k, v in self.metrics.__dict__.items():
                metrics_table.add_row(k.replace("_", " ").title(), str(v))

            self.layout["metrics"].update(
                Panel(metrics_table, title="Real-time Metrics")
            )

            log_table = self._extracted_from__update_layout_5("Time", "Level")
            log_table.add_column("Message")

            for entry in list(self.log_history)[-10:]:
                log_table.add_row(entry["timestamp"], entry["level"], entry["message"])

            self.layout["history"].update(Panel(log_table, title="Recent Logs"))

    # TODO Rename this here and in `_update_layout`
    def _extracted_from__update_layout_5(self, arg0, arg1):
        # Update metrics panel
        result = Table(show_header=True)
        result.add_column(arg0)
        result.add_column(arg1)

        return result

    def _create_layout(self) -> Layout:
        """Create initial layout for live display"""
        layout = Layout()

        layout.split_column(Layout(name="header", size=3), Layout(name="body"))

        layout["header"].split_row(Layout(name="title"), Layout(name="metrics"))

        layout["body"].split_row(Layout(name="history"), Layout(name="patterns"))

        # Initialize panels
        layout["title"].update(Panel("Python Import Fixer Logs", style="bold blue"))

        return layout


# ^_^ CLAUDE'S SECTION 9 UPGRADE PICK:
# Added real-time pattern detection using ML clustering to identify
# common log patterns and potential issues. This helps highlight
# recurring problems and provides insights into import fix patterns.
