"""signature_visitors.py"""

# No typing imports needed here

from src.visitors.visitor import Visitor


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
        """Visit a signature node."""
        self.visit(signature)

    def visit_type(self, type):
        """Visit a type node."""
        self.visit(type)

    def visit_typevar(self, typevar):
        """Visit a typevar node."""
        self.visit(typevar)

    def visit_class(self, class_):
        """Visit a class node."""
        self.visit(class_)

    def visit_array(self, array):
        """Visit an array node."""
        self.visit(array)

    def visit_primitive(self, primitive):
        """Visit a primitive node."""
        self.visit(primitive)

    def visit_method(self, method):
        """Visit a method node."""
        self.visit(method)

    def visit_parameterized(self, parameterized):
        """Visit a parameterized node."""
        self.visit(parameterized)

    def visit_wildcard(self, wildcard):
        """Visit a wildcard node."""
        self.visit(wildcard)

    def visit_bound(self, bound):
        """Visit a bound node."""
        self.visit(bound)

    def visit_annotation(self, annotation):
        """Visit an annotation node."""
        self.visit(annotation)

    def visit_element(self, element):
        """Visit an element node."""
        self.visit(element)

    def visit_name(self, name):
        """Visit a name node."""
        self.visit(name)

    def visit_value(self, value):
        """Visit a value node."""
        self.visit(value)

    def visit_list(self, list):
        """Visit a list node."""
        self.visit(list)

    def visit_map(self, map):
        """Visit a map node."""
        self.visit(map)

    def visit_key(self, key):
        """Visit a key node."""
        self.visit(key)

    def visit_enum(self, enum):
        """Visit an enum node."""
        self.visit(enum)

    def visit_tuple(self, tuple):
        """Visit a tuple node."""
        self.visit(tuple)

    def visit_union(self, union):
        """Visit a union node."""
        self.visit(union)

    def visit_intersection(self, intersection):
        """Visit an intersection node."""
        self.visit(intersection)

    def visit_literal(self, literal):
        """Visit a literal node."""
        self.visit(literal)
