"""
Utilities for generating code.
"""


class CodeWriter:
    """
    Class used to generate formatted text, i.e. text with control of indentation and scope.
    Useful for generating code.
    """

    def __init__(self):
        self.indent_depth = int()
        self.code = str()
        self.scope = list()
        self.reset()
        self.indent = "    "
        self.scope_delimiter = "::"

    def reset(self):
        """Reset code writer."""
        self.indent_depth = 0
        self.code = ""
        self.scope = []

    def empty(self):
        """Check if anything has been written."""
        return self.code == ""

    def print_trace(self, message=""):
        """Insert debug information into code. Useful for tracing the origin of the code."""
        import inspect
        frame = inspect.currentframe().f_back
        func = frame.f_code
        fmt = "// Python debug: %s(%s): %s()"
        values = (func.co_filename, frame.f_lineno, func.co_name)
        my_string = fmt % values
        self.wl(my_string)
        if message != "":
            self.wl("// Python debug msg: \'" + message + "\'")

    def inc_indent(self):
        """Increase indent depth."""
        self.indent_depth += 1

    def dec_indent(self):
        """Decrease indent depth."""
        self.indent_depth -= 1

    def wl(self, string):
        """Write a line of code."""
        if string != "" or not _contains_only_spaces(self.indent):
            self.bl(string)
        self.el("")

    def bl(self, string):
        """Begin a line of code."""
        self.code += self.indent_depth * self.indent
        self.code += string

    def al(self, string):
        """Append code to the current line."""
        self.code += string

    def el(self, string):
        """Append code to the current line and end line."""
        self.code += string + "\n"

    def ws(self):
        """Output current scope."""
        self.code += self.get_scope()

    def wc(self, string):
        """Write several lines of code."""
        # Ignore final line break
        if string[-1] == "\n":
            string = string[:-1]
        for line in string.split("\n"):
            self.wl(line)

    def verbatim(self, string):
        """Write code verbatim."""
        self.code += string

    def get_scope(self):
        """Get current scope."""
        out = ""
        for scope in self.scope:
            out += scope + self.scope_delimiter
        return out

    def pop_first_line(self):
        """Pop and return first line from code."""
        result = self.code.partition("\n")
        self.code = result[2]
        return result[0]

    def count_num_lines(self):
        """
        Count number of lines in the code.

        :return: Number of lines in the code.
        """
        if self.code == "":
            return 0
        return self.code.count("\n") + 1

    def push_scope(self, scope_name):
        """Push scope."""
        self.scope.append(scope_name)

    def pop_scope(self):
        """Pop scope."""
        return self.scope.pop()


def _contains_only_spaces(string):
    """
    Check if a string contains only whitespaces.

    :param string: String to check.
    :return: True/False whether or not the string contains only whitespaces.
    """
    return string.replace(' ', '') == ""
