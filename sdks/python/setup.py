"""
Builder Engine Python SDK - Setup

Build a Voice AI Platform with Builder Engine.
"""

from setuptools import setup, find_packages
import os

# Read the README
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read version
about = {}
with open(os.path.join(here, "builderengine", "__init__.py"), encoding="utf-8") as f:
    exec(f.read(), about)

setup(
    name="builderengine",
    version=about["__version__"],
    author="Builder Engine Team",
    author_email="sdk@builderengine.io",
    description="Python SDK for the Builder Engine AI Voice Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/builderengine/python-sdk",
    project_urls={
        "Documentation": "https://docs.builderengine.io",
        "Changelog": "https://github.com/builderengine/python-sdk/blob/main/CHANGELOG.md",
        "Bug Tracker": "https://github.com/builderengine/python-sdk/issues",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    python_requires=">=3.8",
    install_requires=[
        "httpx>=0.24.0",
    ],
    extras_require={
        "async": [
            "websockets>=11.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "pytest-cov>=4.0",
            "mypy>=1.0",
            "black>=23.0",
            "ruff>=0.0.270",
            "respx>=0.20",
        ],
        "all": [
            "websockets>=11.0",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Communications :: Telephony",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords=[
        "voice",
        "ai",
        "agent",
        "telephony",
        "speech",
        "llm",
        "conversation",
        "builderengine",
        "tts",
        "stt",
        "transcription",
    ],
    package_data={
        "builderengine": ["py.typed"],
    },
    zip_safe=False,
)
