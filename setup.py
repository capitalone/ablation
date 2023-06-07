import os
import re
from pathlib import Path
from typing import List

from setuptools import find_packages, setup

####################################################################################################
# CORE PACKAGE COMPONENTS AND METADATA
####################################################################################################

NAME = "ablation"
PACKAGES = find_packages()
META_PATH = Path("ablation") / "__init__.py"
KEYWORDS = [""]


# fmt: off
PROJECT_URLS = {
    "Documentation": (
        "https://github.com/capitalone/ablation"  # noqa: E501
        "ablation/build/html/index.html"  # noqa: E501
    ),
    "Bug Tracker": "https://github.com/capitalone/ablation/issues",  # noqa: E501
    "Source Code": "https://github.com/capitalone/ablation",  # noqa: E501
}
# fmt: on

CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: Apache Software License",
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
]

# No versioning on extras for dev, always grab the latest
EXTRAS_REQUIRE = {
    # "docs": ["sphinx", "furo", "myst-parser"],
    "tests": [
        "coverage",
        "mypy",
        "pytest==6.2.5",
        "pytest-cov==3.0.0",
        "toml",
    ],
    "qa": [
        "pre-commit",
        "black==22.3.0",
        "mypy",
        "tox",
        "check-manifest",
        "isort",
        "flake8",
        "flake8-docstrings",
        "edgetest",
    ],
    "build": ["twine", "wheel"],
}

EXTRAS_REQUIRE["dev"] = (
    EXTRAS_REQUIRE["tests"]
    # + EXTRAS_REQUIRE["docs"]
    + EXTRAS_REQUIRE["qa"]
    # + EXTRAS_REQUIRE["build"]
)

HERE = Path(__file__).absolute().parent
INSTALL_REQUIRES = (HERE / "requirements.txt").read_text().split("\n")
META_FILE = (HERE / META_PATH).read_text()


def find_meta(meta):
    """Extract __*meta*__ from META_FILE."""
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


VERSION = find_meta("version")
URL = find_meta("url")
LONG = "Ablation studies for evaluating XAI methods"

data_dir = "ablation/data"
data_files = [
    (d, [os.path.join(d, f) for f in files])
    for d, _, files in os.walk(data_dir)
]
####################################################################################################
# Installation functions
####################################################################################################


def install_pkg():
    """Configure the setup for the package."""
    setup(
        name=NAME,
        version=VERSION,
        description=find_meta("description"),
        long_description=LONG,
        url=URL,
        project_urls=PROJECT_URLS,
        author="Samuel Sharpe, Daniel Barcklow, Isha Hameed, Justin Au-Yeung, Brian Barr",
        python_requires=">=3.8.0",
        packages=PACKAGES,
        install_requires=INSTALL_REQUIRES,
        classifiers=CLASSIFIERS,
        extras_require=EXTRAS_REQUIRE,
        include_package_data=True,
        zip_safe=False,
        data_files=data_files,
    )


if __name__ == "__main__":
    install_pkg()
