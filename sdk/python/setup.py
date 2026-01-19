"""
BVRAI Python SDK Setup Configuration.

This allows the SDK to be installed via pip.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bvrai",
    version="1.0.0",
    author="Builder Voice AI",
    author_email="sdk@bvrai.com",
    description="Official Python SDK for the Builder Voice AI Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bvrai/bvrai-python",
    project_urls={
        "Bug Tracker": "https://github.com/bvrai/bvrai-python/issues",
        "Documentation": "https://docs.bvrai.com",
        "Changelog": "https://github.com/bvrai/bvrai-python/blob/main/CHANGELOG.md",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Communications :: Telephony",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Typing :: Typed",
    ],
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=[
        "httpx>=0.25.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
            "black>=23.0.0",
        ],
        "async": [
            "aiohttp>=3.8.0",
        ],
    },
    package_data={
        "bvrai": ["py.typed"],
    },
    keywords=[
        "bvrai",
        "voice",
        "ai",
        "speech",
        "stt",
        "tts",
        "voice-agents",
        "conversational-ai",
    ],
)
