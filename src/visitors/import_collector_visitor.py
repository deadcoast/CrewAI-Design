import ast
import datetime


# Define a class to collect the imports from an AST node
class ImportCollectorVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports = []
        super().__init__()

    def get_imports_from_file(self):
        with open(self, "r") as file:
            tree = ast.parse(file.read())
            visitor = ImportCollectorVisitor()
            visitor.visit(tree)
            return visitor.imports

    def visit_Import(self, node):
        for name in node.names:
            self.imports.append(name.name)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)

        for name in node.names:
            self.imports.append(name.name)
            if name.asname:
                self.imports.append(name.asname)

        if node.level < 0:
            for _ in range(abs(node.level)):
                self.imports.append("..")
                if len(self.imports) > 1 and self.imports[-2] != ".":
                    self.imports.pop(-2)

        if node.level > 0:
            for _ in range(node.level):
                self.imports.append(".")
                if len(self.imports) > 1 and self.imports[-2] != ".":
                    self.imports.pop(-2)

    def datetime_to_timestamp(
        self, year, month, day, hour=0, minute=0, second=0, microsecond=0
    ):
        return datetime.datetime(
            year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None
        )

    def timestamp_to_datetime(self, timestamp):
        return datetime.datetime.fromtimestamp(timestamp)
