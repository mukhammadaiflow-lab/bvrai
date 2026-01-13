"""Builder Engine CLI - Setup configuration."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="builderengine-cli",
    version="1.0.0",
    author="Builder Engine",
    author_email="support@builderengine.io",
    description="Command-line interface for Builder Engine AI Voice Agent Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/builderengine/cli",
    project_urls={
        "Bug Tracker": "https://github.com/builderengine/cli/issues",
        "Documentation": "https://docs.builderengine.io/cli",
        "Source Code": "https://github.com/builderengine/cli",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Communications :: Telephony",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    python_requires=">=3.9",
    install_requires=[
        "click>=8.1.0",
        "httpx>=0.25.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "keyring>=24.0.0",
        "platformdirs>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "builderengine=builderengine_cli.main:cli",
            "bvr=builderengine_cli.main:cli",  # Short alias
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
