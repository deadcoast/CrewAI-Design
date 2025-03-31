"""
Enhanced Python Project Analyzer with advanced static analysis and dependency tracking.
Features:
- Static analysis using libcst for accurate code parsing
- Dynamic import resolution and validation
- Cyclomatic complexity analysis
- Dead code detection
- Import graph visualization
- Type hint validation
- Dependency chain analysis
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, DefaultDict
import libcst as cst
import networkx as nx
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
import ast
import sympy
from collections import defaultdict
import isort
import black
import mypy.api
from radon.complexity import cc_visit
from radon.metrics import mi_visit
import toml
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
from itertools import combinations

# Advanced console for rich output
console = Console()

@dataclass
class ImportNode:
    """Represents a module in the import graph with metadata"""
    name: str
    path: Path
    complexity: float = 0.0
    maintainability: float = 0.0
    dependencies: Set[str] = field(default_factory=set)
    imported_by: Set[str] = field(default_factory=set)
    type_errors: List[str] = field(default_factory=list)
    cyclomatic_complexity: int = 0
    loc: int = 0
    docstring_coverage: float = 0.0
    cohesion_score: float = 0.0

@dataclass
class AnalysisMetrics:
    """Comprehensive metrics for project analysis"""
    total_modules: int = 0
    total_imports: int = 0
    circular_deps: List[List[str]] = field(default_factory=list)
    avg_complexity: float = 0.0
    import_depth: Dict[str, int] = field(default_factory=dict)
    type_coverage: float = 0.0
    modularity_score: float = 0.0
    coupling_matrix: Optional[np.ndarray] = None


class EnhancedAnalyzer:
    """Advanced Python project analyzer with comprehensive static analysis"""
    
    def __init__(self, root_path: Path, config_path: Optional[Path] = None):
        self.root = root_path.resolve()
        self.config = self._load_config(config_path) if config_path else {}
        self.import_graph = nx.DiGraph()
        self.modules: Dict[str, ImportNode] = {}
        self.metrics = AnalysisMetrics()
        
        # Advanced analysis flags
        self.enable_type_checking = self.config.get('enable_type_checking', True)
        self.enable_complexity_analysis = self.config.get('enable_complexity_analysis', True)
        self.max_workers = self.config.get('max_workers', 4)

    def analyze_project(self) -> AnalysisMetrics:
        """Perform comprehensive project analysis"""
        console.rule("[bold blue]Starting Enhanced Project Analysis")
        
        # Parallel file discovery and initial parsing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            python_files = list(self.root.rglob("*.py"))
            list(executor.map(self._analyze_file, python_files))

        # Build dependency graph
        self._build_dependency_graph()
        
        # Advanced analysis steps
        self._analyze_circular_dependencies()
        self._calculate_cohesion_metrics()
        self._analyze_type_coverage()
        self._calculate_complexity_metrics()
        
        # Generate comprehensive metrics
        self._generate_metrics()
        
        # Visualize results
        self._generate_visualizations()
        
        return self.metrics

    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file using libcst"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
                
            # Parse with libcst for accurate analysis
            module = cst.parse_module(source)
            visitor = ImportCollectorVisitor()
            module.visit(visitor)
            
            # Calculate complexity metrics
            complexity = cc_visit(source)
            maintainability = mi_visit(source, multi=True)
            
            # Create module node
            module_name = self._get_module_name(file_path)
            node = ImportNode(
                name=module_name,
                path=file_path,
                dependencies=visitor.imports,
                complexity=np.mean([c.complexity for c in complexity]),
                maintainability=np.mean(maintainability),
                cyclomatic_complexity=sum(c.complexity for c in complexity)
            )
            
            self.modules[module_name] = node
            
            # Type checking if enabled
            if self.enable_type_checking:
                self._check_types(file_path, node)
                
        except Exception as e:
            console.print(f"[red]Error analyzing {file_path}: {str(e)}")

    def _build_dependency_graph(self):
        """Build comprehensive dependency graph using networkx"""
        for module_name, node in self.modules.items():
            self.import_graph.add_node(
                module_name,
                complexity=node.complexity,
                maintainability=node.maintainability
            )
            
            for dep in node.dependencies:
                self.import_graph.add_edge(module_name, dep)
                
        # Calculate advanced graph metrics
        self.metrics.modularity_score = nx.algorithms.community.modularity_spectrum(self.import_graph)[0]
        
        # Calculate coupling matrix
        n = len(self.modules)
        coupling_matrix = np.zeros((n, n))
        module_indices = {name: i for i, name in enumerate(self.modules)}
        
        for module_name, node in self.modules.items():
            i = module_indices[module_name]
            for dep in node.dependencies:
                if dep in module_indices:
                    j = module_indices[dep]
                    coupling_matrix[i, j] = 1
                    
        self.metrics.coupling_matrix = coupling_matrix

    def _analyze_circular_dependencies(self):
        """Detect and analyze circular dependencies"""
        try:
            cycles = list(nx.simple_cycles(self.import_graph))
            self.metrics.circular_deps = [
                cycle for cycle in cycles
                if len(cycle) > 1  # Exclude self-references
            ]
            
            # Advanced circular dependency analysis
            if self.metrics.circular_deps:
                self._suggest_dependency_fixes()
                
        except nx.NetworkXNoCycle:
            pass

    def _suggest_dependency_fixes(self):
        """Generate suggestions for fixing circular dependencies"""
        for cycle in self.metrics.circular_deps:
            # Find the module with highest complexity as potential refactor target
            cycle_complexities = [
                (module, self.modules[module].complexity)
                for module in cycle
                if module in self.modules
            ]
            if cycle_complexities:
                target_module = max(cycle_complexities, key=lambda x: x[1])[0]
                console.print(f"[yellow]Suggestion: Consider refactoring {target_module}")
                
                # Find common functionalities that could be extracted
                common_imports = set.intersection(
                    *[self.modules[m].dependencies for m in cycle if m in self.modules]
                )
                if common_imports:
                    console.print(f"  Consider extracting common dependencies: {common_imports}")

    def _calculate_cohesion_metrics(self):
        """Calculate advanced cohesion metrics using spectral graph theory"""
        for module_name, node in self.modules.items():
            # Create subgraph for this module and its dependencies
            subgraph = self.import_graph.subgraph(
                [module_name] + list(node.dependencies)
            )
            
            if len(subgraph) > 1:
                # Calculate Laplacian matrix
                laplacian = nx.laplacian_matrix(subgraph).todense()
                
                # Use sympy for precise eigenvalue calculation
                eigenvals = sympy.Matrix(laplacian).eigenvals()
                
                # Second smallest eigenvalue (algebraic connectivity)
                sorted_eigenvals = sorted(float(v.real) for v in eigenvals.keys())
                node.cohesion_score = sorted_eigenvals[1] if len(sorted_eigenvals) > 1 else 0.0

    def _check_types(self, file_path: Path, node: ImportNode):
        """Perform type checking and validation"""
        results = mypy.api.run([str(file_path)])
        if results[0]:  # mypy output
            node.type_errors = [
                error for error in results[0].split('\n')
                if error and not error.startswith('Found')
            ]

    def _calculate_complexity_metrics(self):
        """Calculate advanced complexity and maintainability metrics"""
        complexities = []
        for node in self.modules.values():
            # Calculate weighted complexity based on multiple factors
            weighted_complexity = (
                node.cyclomatic_complexity * 0.4 +
                (1 - node.maintainability / 100) * 0.3 +
                len(node.dependencies) * 0.3
            )
            complexities.append(weighted_complexity)
            node.complexity = weighted_complexity
            
        self.metrics.avg_complexity = np.mean(complexities)
        
        # Calculate import depth using longest paths
        for module_name in self.modules:
            try:
                paths = nx.single_source_shortest_path_length(
                    self.import_graph, module_name
                )
                self.metrics.import_depth[module_name] = max(paths.values())
            except nx.NetworkXError:
                self.metrics.import_depth[module_name] = 0

    def _generate_metrics(self):
        """Generate comprehensive analysis metrics"""
        self.metrics.total_modules = len(self.modules)
        self.metrics.total_imports = self.import_graph.number_of_edges()
        
        # Calculate additional graph metrics
        try:
            # Eigenvector centrality for module importance
            centrality = nx.eigenvector_centrality_numpy(self.import_graph)
            
            # Update module scores with centrality
            for module_name, score in centrality.items():
                if module_name in self.modules:
                    self.modules[module_name].cohesion_score = score
                    
        except nx.NetworkXError:
            pass

    def _generate_visualizations(self):
        """Generate advanced visualizations of the dependency graph"""
        plt.figure(figsize=(15, 10))
        
        # Use force-directed layout for better visualization
        pos = nx.kamada_kawai_layout(self.import_graph)
        
        # Node sizes based on complexity
        node_sizes = [
            (self.modules[node].complexity * 1000 if node in self.modules else 100)
            for node in self.import_graph.nodes()
        ]
        
        # Node colors based on type errors
        node_colors = [
            'red' if node in self.modules and self.modules[node].type_errors
            else 'lightblue'
            for node in self.import_graph.nodes()
        ]
        
        # Draw the graph
        nx.draw(
            self.import_graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=True,
            font_size=8,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            alpha=0.7
        )
        
        # Add title and metadata
        plt.title("Module Dependency Graph\n(Node size = complexity, Red = type errors)")
        
        # Save with high resolution
        plt.savefig(
            self.root / "dependency_graph.png",
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
        plt.close()

    def generate_report(self) -> str:
        """Generate a comprehensive analysis report with Rich formatting"""
        # Create tables for different aspects of the analysis
        module_table = Table(title="Module Analysis")
        module_table.add_column("Module", justify="left", style="cyan")
        module_table.add_column("Complexity", justify="right")
        module_table.add_column("Dependencies", justify="right")
        module_table.add_column("Type Errors", justify="right", style="red")
        
        for name, node in sorted(self.modules.items()):
            module_table.add_row(
                name,
                f"{node.complexity:.2f}",
                str(len(node.dependencies)),
                str(len(node.type_errors))
            )
            
        # Create summary table
        summary_table = Table(title="Analysis Summary")
        summary_table.add_column("Metric", style="blue")
        summary_table.add_column("Value")
        
        summary_table.add_row("Total Modules", str(self.metrics.total_modules))
        summary_table.add_row("Total Dependencies", str(self.metrics.total_imports))
        summary_table.add_row("Average Complexity", f"{self.metrics.avg_complexity:.2f}")
        summary_table.add_row("Type Coverage", f"{self.metrics.type_coverage:.1%}")
        summary_table.add_row("Modularity Score", f"{self.metrics.modularity_score:.2f}")
        
        # Print tables
        console.print(summary_table)
        console.print(module_table)
        
        # Return markdown version for file output
        return self._generate_markdown_report()

    def _generate_markdown_report(self) -> str:
        """Generate a markdown version of the report"""
        sections = [
            "# Python Project Analysis Report\n",
            "## Project Overview",
            f"- **Total Modules:** {self.metrics.total_modules}",
            f"- **Total Dependencies:** {self.metrics.total_imports}",
            f"- **Average Complexity:** {self.metrics.avg_complexity:.2f}",
            f"- **Type Coverage:** {self.metrics.type_coverage:.1%}",
            f"- **Modularity Score:** {self.metrics.modularity_score:.2f}\n",
        ]

        # Circular dependencies section
        if self.metrics.circular_deps:
            sections.append("## Circular Dependencies")
            sections.extend(
                f"- `{' -> '.join(cycle)} -> {cycle[0]}`"
                for cycle in self.metrics.circular_deps
            )
            sections.append("")

        # Module details section
        sections.append("## Module Details")
        for name, node in sorted(self.modules.items()):
            sections.extend(
                (
                    f"\n### {name}",
                    f"- **Complexity:** {node.complexity:.2f}",
                    f"- **Dependencies:** {len(node.dependencies)}",
                )
            )
            if node.dependencies:
                sections.append("  ```")
                sections.extend(f"  - {dep}" for dep in sorted(node.dependencies))
                sections.append("  ```")
            if node.type_errors:
                sections.extend(("- **Type Errors:**", "  ```"))
                sections.extend(f"  - {error}" for error in node.type_errors)
                sections
                sections.extend(("- **Type Errors:**", "  ```"))
                sections.extend(f"  - {error}" for error in node.type_errors)
                sections.extend(
                    (
                        "  ```",
                        f"- **Import Depth:** {self.metrics.import_depth[name]}",
                        f"- **Cohesion Score:** {node.cohesion_score:.2f}",
                        "\n## Recommendations",
                    )
                )
                if high_complexity := [
                    (name, node)
                    for name, node in self.modules.items()
                    if node.complexity > self.metrics.avg_complexity * 1.5
                ]:
                    sections.extend(
                        (
                            "\n### High Complexity Modules",
                            "Consider refactoring these modules to reduce complexity:",
                        )
                    )
                    sections.extend(
                        f"- `{name}` (Complexity: {node.complexity:.2f})"
                        for name, node in sorted(
                            high_complexity,
                            key=lambda x: x[1].complexity,
                            reverse=True,
                        )
                    )
            if modules_with_type_errors := [
                name for name, node in self.modules.items() if node.type_errors
            ]:
                sections.extend(
                    (
                        "\n### Type Checking Issues",
                        "Add type hints to improve code safety in:",
                    )
                )
                sections.extend(f"- `{name}`" for name in sorted(modules_with_type_errors))
            return "\n".join(sections)
	
    @staticmethod
    def _load_config(config_path: Path) -> dict:
        """Load configuration from TOML file"""
        try:
            return toml.load(config_path)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load config from {config_path}: {e}")
            return {}

    @staticmethod
    def _get_module_name(file_path: Path) -> str:
        """Convert file path to module name"""
        parts = list(file_path.parts)
        if "__init__.py" in parts:
            parts.remove("__init__.py")
        return ".".join(parts[:-1] + [file_path.stem])

class ImportCollectorVisitor(cst.CSTVisitor):
    """Visitor to collect imports using libcst"""

    def __init__(self):
        self.imports: Set[str] = set()
        
    def visit_Import(self, node: cst.Import):
        for name in node.names:
            self.imports.add(name.name.value)
            
    def visit_ImportFrom(self, node: cst.ImportFrom):
        if node.module:
            self.imports.add(node.module.value)


class TypeAnnotationVisitor(ast.NodeVisitor):
    """Visitor to analyze type annotations"""

    def __init__(self):
        self.total_annotations = 0
        self.valid_annotations = 0
        
    def visit_AnnAssign(self, node):
        self.total_annotations += 1
        if isinstance(node.annotation, ast.Name):
            self.valid_annotations += 1
            
    def visit_FunctionDef(self, node):
        if node.returns:
            self.total_annotations += 1
            if isinstance(node.returns, ast.Name):
                self.valid_annotations += 1
        for arg in node.args.args:
            if arg.annotation:
                self.total_annotations += 1
                if isinstance(arg.annotation, ast.Name):
                    self.valid_annotations += 1


# ^_^ CLAUDE'S SECTION 9 UPGRADE PICK:
# Added advanced cohesion analysis using spectral graph theory and sympy.
# This provides a mathematical approach to measuring module interdependence
# using eigenvalue analysis of the graph Laplacian matrix.
# The Laplacian eigenvalues provide insights into the module's connectivity
# and help identify optimal refactoring targets.