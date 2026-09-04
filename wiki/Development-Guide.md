# Development Guide

This guide outlines the standards and practices for contributing to the project, with a focus on Python development. While most examples and tools mentioned are Python-specific, the underlying principles apply broadly across programming languages.

When developing code, it is essential to follow certain standards to ensure code readability, maintainability, and quality. More info on why this is important can be found [here](https://peps.python.org/pep-0008/). The tools and practices we use for this in Logistiqs are described in this document.

## Docstrings

For documenting our code we use [PyDoc](https://docs.python.org/3/library/pydoc.html). This looks something like this:

```python
def add(a, b):
    """Perform mathematical addition on two given numbers.

    :param a: the first number to add.
    :param b: the second number to add.
    :return: the sum of the two numbers.
    """
    return a + b
```
This way, the documentation is where the code is, making it easy to understand and maintain. PyDoc also allows for automatically exporting and compiling of all documentation into one report, if that ever may be needed. To help enforce consistent docstrings in our code base, we use the style check tools [pydocstyle](http://www.pydocstyle.org/) and [darglint](https://pypi.org/project/darglint/).

## Type Hinting

Type hints clarify the expected input and output of functions, improving both readability and reliability of code. For instance:

```python
def add(a: int, b: int) -> int:
    return a + b
```

With type hints, a type checker like [Mypy](https://mypy-lang.org/) (the one we use) can catch bugs without having to execute the code, reducing the time developers spend on debugging and testing.

## Automated Testing

Manual testing is often labor-intensive and error-prone, especially when verifying that new changes haven't broken existing functionality. Automated tests provide continuous feedback and help catch bugs early. The framework we use for writing automated tests in Python is [PyTest](https://pytest.org/).

When writing tests, we try to follow these principles:
- A test should aim to test one function/behavior instead of many things at once. This helps with quickly finding the bug or issue when a test fails.
- There is no need to test the functioning of external libraries. Isolate your own piece of functionality and focus testing efforts on that.
- Reduce code duplication (using fixtures, parametrized tests in PyTest). This prevents cumbersome test updating when code/interfaces are changed, affecting tests.
- Write strong tests rather than weak tests. To be more precise: a weak test may only check if code doesn't crash or that a function returns the correct type, a strong test has concrete inputs and expected outputs to check function behavior against.

## Code Formatting and Linting

We enforce consistent code style using [black](https://pypi.org/project/black/) for automatic formatting and [flake8](https://flake8.pycqa.org/) for linting. These tools help maintain readable and uniform code, in line with [PEP 8](https://peps.python.org/pep-0008/) recommendations.

Note: this branch is trimmed to reproducing the IEEE paper's experiments and does not carry the pre-commit hooks or CI pipeline used on `research`/`dev` — see those branches for the full development setup.
