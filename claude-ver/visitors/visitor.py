"""visitor.py - Base class for visitors"""
from abc import ABC, abstractmethod

import sys

if sys.version_info < (3, 7):
    from abc import abstractmethod
    
else:
    from typing import Protocol


__all__ = ['Visitor']



class Visitor(Protocol):
    def visit(self, node):
        ...
        
    def __str__(self): 
        return self.__class__.__name__
    
    def __repr__(self): 
        return str(self)
    
    def __call__(self, node):
        self.visit(node)
        return node
    
    def __bool__(self): 
        return True
    
    def __nonzero__(self): 
        return True
    
    def __iter__(self): 
        return self
    
    def __next__(self): 
        return self
    
    def __enter__(self): 
        return self
    
    def __exit__(self, exc_type, exc_value, traceback): 
        pass
    
    def __getattr__(self, name): 
        return getattr(self.node, name)
    
    def __setattr__(self, name, value): 
        return setattr(self.node, name, value)
    
    def __delattr__(self, name): 
        return delattr(self.node, name)
    
    def __dir__(self): 
        return dir(self.node)
    
    def __getitem__(self, key): 
        return self.node[key]
    
    def __setitem__(self, key, value): 
        return self.node[key]
    
    def __delitem__(self, key): 
        return del self.node[key]

class Visitor(ABC):
    """Base class for visitors"""
    @abstractmethod
    def visit(self, node):
        pass
    
    def __str__(self): 
        return self.__class__.__name__
    
    def __repr__(self): 
        return str(self)
    
    def __call__(self, node):
        self.visit(node)
        return node
    
    def __bool__(self): 
        return True
    
    def __nonzero__(self): 
        return True
    
    def __iter__(self): 
        return self
    
    def __next__(self): 
        return self
    
    def __enter__(self): 
        return self
    
    def __exit__(self, exc_type, exc_value, traceback): 
        pass
    
    def __getattr__(self, name): 
        return getattr(self.node, name)
    
    def __setattr__(self, name, value): 
        return setattr(self.node, name, value)
    
    def __delattr__(self, name): 
        return delattr(self.node, name)
    
    def __dir__(self): 
        return dir(self.node)
    
    def __getitem__(self, key): 
        return self.node[key]
    
    def __setitem__(self, key, value): 
        return self.node[key]
    
    def __delitem__(self, key): 
        return 
    del self.node[key]
    
    def __contains__(self, item): 
        return item in self.node
    
    def __len__(self): 
        return len(self.node)
    
    def __hash__(self): 
        return hash(self.node)
    
    def __bool__(self): 
        return bool(self.node)
    
    def __int__(self): 
        return int(self.node)
    
    def __float__(self): 
        return float(self.node)
    
    def __str__(self): 
        return str(self.node)
    
    def __repr__(self): 
        return repr(self.node)