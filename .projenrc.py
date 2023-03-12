from projen.python import PythonProject

project = PythonProject(
    author_email="hahaschool.wing@gmail.com",
    author_name="Adam Zhang",
    module_name="SAKURA",
    name="SAKURA",
    version="0.1.0",
    pip=False,
    venv=False,
    license=None,
    setuptools=False,
    pytest=True,
    poetry=True,
    poetry_options={
        "repository": "https://github.com/hahaschool/SAKURA.git",
    },
    deps=[
        "python@~3.9",
        "loguru@^0.6.0",
        "tqdm@^4.65.0",
        "numpy@^1.24.2",
        "torch@^1.13.1",
        "tensorboardX@^2.6",
        "pandas@^1.5.3",
        "scikit-learn@^1.2.2",
        "scipy@^1.10.1",
    ],
    dev_deps=["pytest@^6.2.5", "pytest-asyncio@^0.16.0"],
    github_options={
        "pull_request_lint_options": {
            "semantic_title_options": {"types": ["feat", "fix", "chore"]}
        }
    },
)

project.synth()
