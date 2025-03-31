"""signature_visitors.py"""

import ast
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from libcst import (
    Array,
    Class,
    Enum,
    Intersection,
    List,
    Literal,
    Map,
    Name,
    Parameter,
    Parameterized,
    Primitive,
    Signature,
    Tuple,
    Type,
    TypeVar,
    Union,
    Visitor,
    Wildcard,
    WildcardElement,
)

from visitors.visitor import Visitor

class SignatureVisitor(Visitor):
    def visit(self, signature):
        self.visit_signature(signature)
        self.visit_types(signature.types)
        self.visit_parameters(signature.parameters)
        self.visit_returntype(signature.returntype)
        self.visit_exceptions(signature.exceptions)
        self.visit_generics(signature.generics)
        self.visit_varargs(signature.varargs)
        self.visit_kwargs(signature.kwargs)
        self.visit_annotations(signature.annotations)
        self.visit_typevars(signature.typevars)
        self.visit_class(signature.class_)
        self.visit_array(signature.array)
        self.visit_primitive(signature.primitive)
        self.visit_method(signature.method)
        self.visit_parameterized(signature.parameterized)
        self.visit_wildcard(signature.wildcard)
        self.visit_bound(signature.bound)
        self.visit_annotation(signature.annotation)
        self.visit_element(signature.element)
        self.visit_name(signature.name)
        self.visit_value(signature.value)
        self.visit_list(signature.list)
        self.visit_map(signature.map)
        self.visit_key(signature.key)
        self.visit_value(signature.value)
        self.visit_enum(signature.enum)
        self.visit_tuple(signature.tuple)
        self.visit_union(signature.union)
        self.visit_intersection(signature.intersection)
        self.visit_literal(signature.literal)
        
    def visit_signature(self, signature):
        self.visit(signature)

    def visit_type(self, type):
        self.visit(type)

    def visit_typevar(self, typevar):
        self.visit(typevar)

    def visit_class(self, class_):
        self.visit(class_)

    def visit_array(self, array):
        self.visit(array)

    def visit_primitive(self, primitive):
        self.visit(primitive)

    def visit_method(self, method):
        self.visit(method)
        
    def visit_parameterized(self, parameterized):
        self.visit(parameterized)
        
    def visit_wildcard(self, wildcard):
        self.visit(wildcard)
        
    def visit_bound(self, bound):
        self.visit(bound)
        
    def visit_annotation(self, annotation):
        self.visit(annotation)
        
    def visit_element(self, element):
        self.visit(element)
        
    def visit_name(self, name):
        self.visit(name)
        
    def visit_value(self, value):
        self.visit(value)
        
    def visit_list(self, list):
        self.visit(list)
        
    def visit_map(self, map):
        self.visit(map)
        
    def visit_key(self, key):
        self.visit(key)
        
    def visit_value(self, value):
        self.visit(value)
        
    def visit_enum(self, enum):
        self.visit(enum)
         
    def visit_tuple(self, tuple):
        self.visit(tuple)
        
    def visit_union(self, union):
        self.visit(union)
        
    def visit_intersection(self, intersection):
        self.visit(intersection)
        
    def visit_literal(self, literal):
        self.visit(literal)
        