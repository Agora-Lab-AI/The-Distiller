from distutils.core import setup
from setuptools import find_packages

with open("README.md", "r", encoding = "utf-8") as readme:
    long_description = readme.read()

setup(
    name="The Distiller",
    version="0.0.2",
    description="Generate textual and conversational datasets with LLMs.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    author="Kye Gomez",
    author_email="Kye@apac.ai",
    url="https://github.com/kyegomez/The-Distiller",
    keywords=["dataset", "llm", "langchain", "openai"],
    package_dir={"": "src"},
    packages = find_packages(where="src"),
    install_requires=[
        "langchain>=0.0.113",
        "click>=8.1"
    ],
    entry_points={
        "console_scripts": [
            "distiller=distiller:distiller"
        ],
    },
)
