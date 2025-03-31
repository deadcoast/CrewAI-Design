"""
Enhanced Python code signature analyzer with advanced type inference and analysis.
Features:
- Deep signature analysis using advanced AST traversal
- Type inference using machine learning and pattern matching
- Function complexity and cohesion metrics
- Signature clustering and similarity analysis
- Advanced visualization capabilities
- Import dependency tracking with graph theory
"""

from __future__ import annotations

import ast
import contextlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union, TypeVar, Generic
import libcst as cst
from pydantic import BaseModel, Field
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import networkx as nx
import numpy as np
from rich.console import Console
from rich.tree import Tree
from rich.syntax import Syntax
import typeguard
from typing_extensions import Protocol, runtime_checkable
from collections import defaultdict
import rustworkx as rx
import sympy
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import torch
import torch.nn.functional as F

console = Console()

T = TypeVar('T')

@runtime_checkable
class Callable(Protocol):
    """Protocol for callable objects with signature information"""
    __signature__: inspect.Signature

@dataclass
class TypeInfo:
    """Enhanced type information with inference confidence"""
    type_hint: Optional[str]
    inferred_type: Optional[str]
    confidence: float
    source_locations: Set[str] = field(default_factory=set)
    constraints: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Validate type consistency"""
        if self.type_hint and self.inferred_type:
            try:
                typeguard.check_type("", eval(self.inferred_type), eval(self.type_hint))
            except Exception:
                self.confidence *= 0.5

class SignatureMetrics(BaseModel):
    """Advanced metrics for code signatures"""
    complexity: float = Field(0.0, ge=0.0, le=1.0)
    cohesion: float = Field(0.0, ge=0.0, le=1.0)
    coupling: float = Field(0.0, ge=0.0, le=1.0)
    maintainability: float = Field(0.0, ge=0.0, le=1.0)
    type_safety: float = Field(0.0, ge=0.0, le=1.0)
    documentation_score: float = Field(0.0, ge=0.0, le=1.0)

@dataclass
class SignatureComponent:
    """Component of a signature with enhanced analysis"""
    name: str
    type_info: TypeInfo
    default_value: Optional[str] = None
    is_optional: bool = False
    constraints: List[str] = field(default_factory=list)
    usage_locations: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'type': self.type_info.type_hint or self.type_info.inferred_type,
            'confidence': self.type_info.confidence,
            'optional': self.is_optional,
            'constraints': self.constraints
        }

@dataclass
class CodeSignature:
    """Enhanced code signature with comprehensive analysis"""
    name: str
    module_path: Path
    components: List[SignatureComponent]
    return_type: Optional[TypeInfo] = None
    docstring: Optional[str] = None
    metrics: SignatureMetrics = field(default_factory=SignatureMetrics)
    dependencies: Set[str] = field(default_factory=set)
    call_graph: Optional[nx.DiGraph] = None
    
    def similarity_score(self, other: CodeSignature) -> float:
        """Calculate signature similarity using TF-IDF and cosine similarity"""
        vectorizer = TfidfVectorizer()
        signatures = [
            f"{self.name} {' '.join(c.name for c in self.components)}",
            f"{other.name} {' '.join(c.name for c in other.components)}"
        ]
        tfidf_matrix = vectorizer.fit_transform(signatures)
        return 1 - cosine(tfidf_matrix.toarray()[0], tfidf_matrix.toarray()[1])

