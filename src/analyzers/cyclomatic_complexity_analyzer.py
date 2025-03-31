"""cyclomatic_complexity_analyzer class"""

from pathlib import Path
from typing import List, Optional, Set

import libcst as cst
import networkx as nx
import numpy as np
from collectors.import_collector_visitor import ImportCollectorVisitor


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
        return f"ImportNode({self.name}, {self.path}, {self.dependencies}, {self.complexity}, {self.maintainability}, {self.cyclomatic_complexity})"

    def __repr__(self):
        return f"ImportNode({self.name}, {self.path}, {self.dependencies}, {self.complexity}, {self.maintainability}, {self.cyclomatic_complexity})"

    def __hash__(self):
        return hash(
            (
                self.name,
                self.path,
                self.dependencies,
                self.complexity,
                self.maintainability,
                self.cyclomatic_complexity,
            )
        )

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.path == other.path
            and self.dependencies == other.dependencies
        )

    def __lt__(self, other):
        return self.name < other.name

    def __le__(self, other):
        return self.name <= other.name

    def __gt__(self, other):
        return self.name > other.name


class CyclomaticComplexityAnalyzer:
    def __init__(self):
        self.import_graph = {}
        self.modules = {}
        self.metrics = {}

        # Advanced analysis flags
        self.enable_type_checking = True
        self.enable_complexity_analysis = True
        self.max_workers = 4

        self.import_graph = nx.DiGraph()
        self.modules = {}
        self.metrics = {}

        # Advanced analysis flags
        self.enable_type_checking = True
        self.enable_complexity_analysis = True
        self.max_workers = 4

    def analyze_project(self, root_path: Path, config_path: Optional[Path] = None):
        self.root = root_path.resolve()
        self.config = self._load_config(config_path) if config_path else {}
        self.import_graph = nx.DiGraph()
        self.modules = {}
        self.metrics = {}

        # Advanced analysis flags
        self.enable_type_checking = self.config.get("enable_type_checking", True)
        self.enable_complexity_analysis = self.config.get(
            "enable_complexity_analysis", True
        )
        self.max_workers = self.config.get("max_workers", 4)

        self.import_graph = nx.DiGraph()

        return self.import_graph

    def analyze_file(self, file_path: Path):
        self._analyze_file(file_path)
        return self.import_graph

    def _analyze_file(self, file_path: Path):
        with open(file_path, "r") as file:
            code = file.read()
            tree = cst.parse_module(code)
            visitor = ImportCollectorVisitor()
            visitor.visit(tree)
            self.import_graph[file_path] = visitor.imports

        # Type checking if enabled
        if self.enable_type_checking:
            self.type_errors = self._check_types(
                file_path, self.import_graph[file_path]
            )
        else:
            self.type_errors = []

        # Cyclomatic complexity analysis if enabled
        if self.enable_complexity_analysis:
            self.modules = self._analyze_complexity(file_path)
            self.metrics = self._calculate_complexity_metrics()
        else:
            self.modules = {}
            self.metrics = {}

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

        # Generate reports in different formats
        self._generate_reports()

        # Suggest fixes for issues identified during analysis
        self._suggest_fixes()

        return self.import_graph

    def _calculate_complexity_metrics(self):
        self.metrics = {}
        for module_name, module_data in self.modules.items():
            self.metrics[module_name] = {
                "complexity": module_data["complexity"],
                "maintainability": module_data["maintainability"],
            }

        return self.metrics

    def _calculate_cohesion_metrics(self):
        self.cohesion_metrics = {}

        for module_name, module_data in self.modules.items():
            self.cohesion_metrics[module_name] = {
                "cohesion": module_data["cohesion"],
                "complexity": module_data["complexity"],
                "maintainability": module_data["maintainability"],
            }

        return self.cohesion_metrics

    def _generate_metrics(self):
        self.metrics = {
            "total_modules": len(self.modules),
            "total_imports": self.import_graph.number_of_edges(),
        }

        # Calculate additional graph metrics
        self.metrics["eigenvector_centrality"] = nx.eigenvector_centrality_numpy(
            self.import_graph
        )
        self.metrics["average_cohesion"] = np.mean(
            [module["cohesion"] for module in self.cohesion_metrics.values()]
        )
        self.metrics["average_eigenvector_centrality"] = np.mean(
            list(self.metrics["eigenvector_centrality"].values())
        )
        self.metrics["average_complexity"] = np.mean(
            [module["complexity"] for module in self.modules.values()]
        )
        self.metrics["average_maintainability"] = np.mean(
            [module["maintainability"] for module in self.modules.values()]
        )

        # Calculate advanced metrics
        self.metrics["average_cyclomatic_complexity"] = np.mean(
            [module["cyclomatic_complexity"] for module in self.modules.values()]
        )
        self.metrics["average_loc"] = np.mean(
            [module["loc"] for module in self.modules.values()]
        )
        self.metrics["average_sloc"] = np.mean(
            [module["sloc"] for module in self.modules.values()]
        )
        self.metrics["average_lloc"] = np.mean(
            [module["lloc"] for module in self.modules.values()]
        )

        # Calculate additional metrics
        self.metrics["total_type_errors"] = len(self.type_errors)
        self.metrics["type_errors"] = self.type_errors
        self.metrics["cyclic_dependencies"] = self.cyclic_deps
        self.metrics["suggestions"] = self.suggestions

        return self.metrics

    def _check_types(self, file_path: Path, node: ImportNode):
        # Perform type checking and validation
        # ... (implementation details omitted for brevity)

        return self.type_errors

    def _analyze_type_coverage(self, file_path: Path):
        # Perform type checking and validation
        # ... (implementation details omitted for brevity)

        return self.type_errors

    def _analyze_complexity(self, file_path: Path):
        # Perform cyclomatic complexity analysis
        # ... (implementation details omitted for brevity)

        return self.modules

    def _build_dependency_graph(self):
        # Build the dependency graph
        # ... (implementation details omitted for brevity)

        return self.import_graph

    def _analyze_circular_dependencies(self):
        # Detect and analyze circular dependencies
        # ... (implementation details omitted for brevity)

        return self.cyclic_deps

    def _suggest_dependency_fixes(self):
        # Generate suggestions for fixing circular dependencies
        # ... (implementation details omitted for brevity)

        return self.suggestions

    def _calculate_cohesion_metrics(self):
        # Calculate advanced cohesion metrics using spectral graph theory
        # ... (implementation details omitted for brevity)

        return self.cohesion_metrics

    def _calculate_complexity_metrics(self):
        # Calculate advanced complexity and maintainability metrics
        # ... (implementation details omitted for brevity)

        return self.complexity_metrics

    def _calculate_import_depth(self):
        # Calculate import depth using longest paths
        # ... (implementation details omitted for brevity)

        return self.import_depth

    def _calculate_maintainability_index(self):
        # Calculate maintainability index based on cyclomatic complexity
        # ... (implementation details omitted for brevity)

        return self.maintainability

    def _calculate_modularity_score(self):
        # Calculate modularity score using spectral graph theory
        # ... (implementation details omitted for brevity)

        return self.modularity

    def _calculate_type_coverage(self):
        # Calculate type coverage metrics
        # ... (implementation details omitted for brevity)

        return self.type_coverage

    def _generate_markdown_report(self):
        # Generate a markdown version of the report
        # ... (implementation details omitted for brevity)

        return self.markdown_report

    def _generate_html_report(self):
        # Generate an HTML version of the report
        # ... (implementation details omitted for brevity)

        return self.html_report

    def _load_config(self, config_path: Optional[Path] = None):
        # Load configuration from file
        # ... (implementation details omitted for brevity)

        return self.config

    def _save_report(self, report_path: Path):
        # Save the report to a file
        # ... (implementation details omitted for brevity)

        return
