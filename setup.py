from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    requirements_path = Path(__file__).parent / "requirements.txt"
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


setup(
    name="mlops-end-to-end-pipeline",
    version="1.0.0",
    packages=find_packages(),
    install_requires=read_requirements(),
    author="Your Name",
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "mlops-pipeline=src.pipeline:main",
        ],
    },
)
