"""visitor.py - Base class for visitors"""

from typing import Protocol

__all__ = ["Visitor"]


class Visitor(Protocol):
    def visit(self, node):
        """Visit a node."""

    def __str__(self):
        """Return the name of the visitor."""
        return self.__class__.__name__

    def __repr__(self):
        """Return the name of the visitor."""
        return str(self)

    def __call__(self, node):
        self.visit(node)
        return node

    def __bool__(self):
        """Return True."""
        return True

    def __nonzero__(self):
        """Return True."""
        return True

    def __iter__(self):
        """Return self."""
        return self

    def __next__(self):
        """Return self."""
        return self

    def __enter__(self):
        """Return self."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Do nothing."""
        pass

    def __getattr__(self, name):
        """Return the attribute of the node."""
        return getattr(self.node, name)

    def __setattr__(self, name, value):
        """Set the attribute of the node."""
        return setattr(self.node, name, value)

    def __delattr__(self, name):
        """Delete the attribute of the node."""
        return delattr(self.node, name)

    def __dir__(self):
        """Return the attributes of the node."""
        return dir(self.node)

    def __getitem__(self, key):
        """Return the item of the node."""
        return self.node[key]

    def __setitem__(self, key, value):
        """Set the item of the node."""
        return self.node[key]

    def __delitem__(self, key):
        """Delete the item of the node."""
        del self.node[key]

    def __contains__(self, item):
        """Return True if the item is in the node."""
        return item in self.node

    def __len__(self):
        """Return the length of the node."""
        return len(self.node)

    def __hash__(self):
        """Return the hash of the node."""
        return hash(self.node)

    def __int__(self):
        """Return the integer value of the node."""
        return int(self.node)

    def __float__(self):
        """Return the float value of the node."""
        return float(self.node)
