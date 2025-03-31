"""import_analyzer.py: Advanced import resolver and validator"""

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set

import toml
import libcst as cst
import mypy.api
import networkx as nx
import numpy as np
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from rich.console import Console
from rich.table import Table

# Local imports
from .analyzer import EnhancedAnalyzer, AnalysisMetrics
from .cyclomatic_complexity_analyzer import CyclomaticComplexityAnalyzer

console = Console()


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


class ImportNode:
    def __init__(
        self,
        name: str,
        path: Path,
        dependencies: Set[str],
        complexity: float,
        maintainability: float,
        cyclomatic_complexity: int,
    ):
        self.name = name
        self.path = path
        self.dependencies = dependencies
        self.complexity = complexity
        self.maintainability = maintainability
        self.cyclomatic_complexity = cyclomatic_complexity
        self.imported_by: Set[str] = set()
        self.type_errors: List[str] = []
        self.cyclomatic_complexity = 0
        self.loc = 0
        self.docstring_coverage = 0.0
        self.cohesion_score = 0.0

    def __str__(self):
        return (
            f"Module: {self.name}, Path: {self.path}, Dependencies: {self.dependencies}"
        )

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    def __le__(self, other):
        return self.name <= other.name

    def __gt__(self, other):
        return self.name > other.name

    def __ge__(self, other):
        return self.name >= other.name

    def __ne__(self, other):
        return self.name != other.name

    def __len__(self):
        return len(self.dependencies)

    def __iter__(self):
        return iter(self.dependencies)

    def __contains__(self, item):
        return item in self.dependencies

    def __add__(self, other):
        return self.dependencies.union(other.dependencies)

    def __iadd__(self, other):
        self.dependencies.update(other.dependencies)
        return self

    def __sub__(self, other):
        return self.dependencies.difference(other.dependencies)

    def __isub__(self, other):
        self.dependencies.difference_update(other.dependencies)
        return self

    def __or__(self, other):
        return self.dependencies | other.dependencies

    def __ior__(self, other):
        self.dependencies |= other.dependencies
        return self

    def __and__(self, other):
        return self.dependencies & other.dependencies

    def __iand__(self, other):
        self.dependencies &= other.dependencies
        return self

    def __xor__(self, other):
        return self.dependencies ^ other.dependencies

    def __ixor__(self, other):
        self.dependencies ^= other.dependencies
        return self


class ImportAnalyzer:
    """Advanced import resolver and validator"""

    def __init__(self):
        self.import_graph = nx.DiGraph()
        self.modules: Dict[str, ImportNode] = {}
        self.imports: Set[str] = set()
        self.imported_by: DefaultDict[str, Set[str]] = defaultdict(set)
        self.type_errors: DefaultDict[str, List[str]] = defaultdict(list)
        self.cyclic_deps: List[List[str]] = []
        self.depth_map: Dict[str, int] = {}
        self.coupling_matrix: Optional[np.ndarray] = None

        # Advanced analysis flags
        self.enable_type_checking = True
        self.enable_complexity_analysis = True
        self.max_workers = 4

    def analyze_project(self, root_path: Path, config_path: Optional[Path] = None):
        """Perform advanced import analysis on a Python project"""
        analyzer = EnhancedAnalyzer(root_path, config_path)
        return analyzer.analyze_project()

    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file using libcst"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
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
                cyclomatic_complexity=sum(c.complexity for c in complexity),
            )

            self.modules[module_name] = node

            # Type checking if enabled
            if self.enable_type_checking:
                self._analyze_type_coverage(file_path)

            # Complexity analysis if enabled
            if self.enable_complexity_analysis:
                self._analyze_complexity(file_path)

        except Exception as e:
            console.print(f"[red]Error analyzing {file_path}: {str(e)}")

        return node

    def _get_module_name(self, file_path: Path):
        """Extract module name from file path"""
        return file_path.stem.replace("_", "-").lower().replace("-", "_")

    def _analyze_type_coverage(self, file_path: Path):
        """Perform type checking and validation"""
        results = mypy.api.run([str(file_path)])
        if results[0]:  # mypy output
            self.type_errors[file_path] = [
                error
                for error in results[0].split("\n")
                if error and not error.startswith("Found")
            ]

        return self.type_errors

    def _analyze_complexity(self, file_path: Path):
        """Perform cyclomatic complexity analysis"""
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        complexity = cc_visit(source)
        self.modules[self._get_module_name(file_path)].cyclomatic_complexity = sum(
            c.complexity for c in complexity
        )
        return self.modules

    def _build_dependency_graph(self):
        for module_name, node in self.modules.items():
            for dependency in node.dependencies:
                self.import_graph.add_edge(module_name, dependency)

        return self.import_graph

    def _calculate_complexity_metrics(self):
        complexities = []
        for node in self.modules.values():
            # Calculate weighted complexity based on multiple factors
            weighted_complexity = (
                node.cyclomatic_complexity * 0.4
                + (1 - node.maintainability / 100) * 0.3
                + len(node.dependencies) * 0.3
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
                self.depth_map[module_name] = max(paths.values())
            except nx.exception.NetworkXNoPath:
                self.depth_map[module_name] = 0

        return self.depth_map

    def _find_cyclic_dependencies(self):
        self.cyclic_deps = list(nx.simple_cycles(self.import_graph))
        return self.cyclic_deps

    def _suggest_dependency_fixes(self):
        for cycle in self.cyclic_deps:
            if len(cycle) > 1:  # Exclude self-references
                cycle = list(cycle)
                module = cycle[0]
                for i in range(1, len(cycle)):
                    dep = cycle[i]
                    console.print(
                        f"[yellow]Suggestion: Consider refactoring {module} to remove dependency on {dep}"
                    )

                    # Find common functionalities that could be extracted
                    common_imports = set.intersection(
                        self.modules[module].dependencies,
                        self.modules[dep].dependencies,
                    )
                    if common_imports:
                        console.print(
                            f"[yellow]Common imports: {', '.join(common_imports)}"
                        )
                    else:
                        console.print("[yellow]No common imports found.")

                    # Suggest potential refactorings
                    refactorings = [
                        "Extract common functionality into a new module",
                        "Move common functionality into a new module",
                        "Extract common functionality into a new class",
                        "Move common functionality into a new class",
                        "Extract common functionality into a new function",
                        "Move common functionality into a new function",
                        "Extract common functionality into a new method",
                        "Move common functionality into a new method",
                    ]
                    console.print(
                        f"[yellow]Suggested refactorings: {', '.join(refactorings)}"
                    )

        return self.cyclic_deps

    def _calculate_modularity_score(self):
        self.modularity_score = self.metrics.type_coverage * self.metrics.avg_complexity
        return self.modularity_score

    def generate_report(self):
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
                str(len(node.type_errors)),
            )

        # Print tables
        console.print(module_table)

        # Print other metrics
        console.print(f"Average Complexity: {self.metrics.avg_complexity:.2f}")
        console.print(f"Type Coverage: {self.metrics.type_coverage:.2f}%")
        console.print(f"Modularity Score: {self.metrics.modularity_score:.2f}")

        # Print suggestions for refactoring
        console.print("[yellow]Suggestions for Refactoring:")
        for cycle in self.cyclic_deps:
            console.print(f"[yellow]Cycle: {', '.join(cycle)}")

            # Find common functionalities that could be extracted
            common_imports = set.intersection(
                *[self.modules[m].dependencies for m in cycle]
            )
            if common_imports:
                console.print(f"[yellow]Common imports: {', '.join(common_imports)}")
            else:
                console.print("[yellow]No common imports found.")

            # Suggest potential refactorings
            refactorings = [
                "Extract common functionality into a new module",
                "Move common functionality into a new module",
                "Extract common functionality into a new class",
                "Move common functionality into a new class",
                "Extract common functionality into a new function",
                "Move common functionality into a new function",
                "Extract common functionality into a new method",
                "Move common functionality into a new method",
            ]
            console.print(f"[yellow]Suggested refactorings: {', '.join(refactorings)}")

        return self.metrics

    def _load_config(self, config_path: Path) -> dict:
        try:
            return toml.load(config_path)
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not load config from {config_path}: {e}"
            )
        return self.config

    def _initialize_complexity_analysis(self):
        self.metrics = AnalysisMetrics()
        self.depth_map = {}
        self.cyclic_deps = []
        self.import_graph = nx.DiGraph()
        self.modules: Dict[str, ImportNode] = {}
        self.metrics = AnalysisMetrics()

        # Advanced analysis flags
        self.enable_type_checking = self.config.get("enable_type_checking", True)
        self.enable_complexity_analysis = self.config.get(
            "enable_complexity_analysis", True
        )
        self.max_workers = self.config.get("max_workers", 4)

        self.import_graph = nx.DiGraph()

        return self.import_graph

    def _analyze_file(self, file_path: Path):
        analyzer = ImportAnalyzer()
        analyzer.analyze_file(file_path)
        self.import_graph = analyzer.import_graph
        self.modules.update(analyzer.modules)
        self.metrics = analyzer.metrics

        return self.import_graph

    def _build_dependency_graph(self):
        for module_name, node in self.modules.items():
            self.import_graph.add_node(
                module_name,
                complexity=node.complexity,
                maintainability=node.maintainability,
            )

            for dep in node.dependencies:
                self.import_graph.add_edge(module_name, dep)

        # Calculate advanced graph metrics
        self.metrics.modularity_score = nx.algorithms.community.modularity_spectrum(
            self.import_graph
        )[0]

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

        return self.import_graph

    def _find_cyclic_dependencies(self):
        self.cyclic_deps = list(nx.simple_cycles(self.import_graph))
        return self.cyclic_deps

    def _suggest_dependency_fixes(self):
        for cycle in self.cyclic_deps:
            console.print(f"[yellow]Cycle: {', '.join(cycle)}")

            # Find common functionalities that could be extracted
            common_imports = set.intersection(
                *[self.modules[m].dependencies for m in cycle]
            )
            if common_imports:
                console.print(f"[yellow]Common imports: {', '.join(common_imports)}")
            else:
                console.print("[yellow]No common imports found.")

            # Suggest potential refactorings
            refactorings = [
                "Extract common functionality into a new module",
                "Move common functionality into a new module",
                "Extract common functionality into a new class",
                "Move common functionality into a new class",
                "Extract common functionality into a new function",
                "Move common functionality into a new function",
                "Extract common functionality into a new method",
                "Move common functionality into a new method",
            ]
            console.print(f"[yellow]Suggested refactorings: {', '.join(refactorings)}")

        return self.metrics

    def _calculate_complexity_metrics(self):
        for module in self.modules:
            self.depth_map[module] = self._calculate_module_depth(module)
            self.modules[module].complexity = self._calculate_complexity(module)

        return self.modules

    def _calculate_module_depth(self, module):
        if module not in self.modules:
            return 0
        return 1 + max(
            self._calculate_module_depth(dep)
            for dep in self.modules[module].dependencies
        )

    def _calculate_complexity(self, module):
        if module not in self.modules:
            return 0
        return sum(
            self._calculate_complexity(dep) for dep in self.modules[module].dependencies
        )

    def _analyze_complexity(self, file_path: Path):
        analyzer = CyclomaticComplexityAnalyzer()
        analyzer.analyze_file(file_path)
        self.import_graph = analyzer.import_graph
        self.modules.update(analyzer.modules)
        self.metrics = analyzer.metrics

        return self.import_graph
