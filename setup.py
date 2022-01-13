from pathlib import Path

from setuptools import setup, find_packages

try:
    from pygpcca import __email__, __author__, __version__, __maintainer__
except ImportError:
    __author__ = __maintainer__ = "Bernhard Reuter"
    __version__ = "1.0.3"
    __email__ = "bernhard-reuter@gmx.de"

setup(
    name="pygpcca",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    version=__version__,
    author=__author__,
    author_email=__email__,
    maintainer=__maintainer__,
    maintainer_email=__email__,
    description=Path("README.rst").read_text("utf-8").splitlines()[34],
    long_description="\n".join(Path("README.rst").read_text("utf-8").splitlines()[:-4]).replace("|br|", "\n"),
    long_description_content_type="text/x-rst; charset=UTF-8",
    url="https://github.com/msmdev/pygpcca",
    download_url="https://pypi.org/project/pygpcca/",
    project_urls={
        "Documentation": "https://pygpcca.readthedocs.io/en/latest",
        "Source Code": "https://github.com/msmdev/pygpcca",
    },
    license="LGPLv3+",
    platforms=["Linux", "MacOSX"],
    packages=find_packages(),
    zip_safe=False,
    install_requires=[line.strip() for line in Path("requirements.txt").read_text("utf-8").splitlines()],
    extras_require=dict(
        # https://gitlab.com/petsc/petsc/-/issues/803
        slepc=[
            "mpi4py>=3.0.3",
            "petsc>=3.13,!=3.14",
            "slepc>=3.13,!=3.14",
            "petsc4py>=3.13,!=3.14",
            "slepc4py>=3.13,!=3.14",
        ],
        dev=["pre-commit>=2.9.0", "bump2version"],
        test=["tox>=3.20.1"],
        docs=[
            line.strip()
            for line in (Path("docs") / "requirements.txt").read_text("utf-8").splitlines()
            if not line.startswith("-r")
        ],
    ),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Typing :: Typed",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=sorted(
        {
            "GPCCA",
            "G-PCCA",
            "Generalized Perron Cluster Cluster Analysis",
            "Markov state model",
            "Markov state modeling",
            "coarse-graining",
            "spectral clustering",
            "non-equilibrium system",
            "non-reversible process",
            "non-autonomous process",
            "cyclic states",
            "metastable states",
            "molecular dynamics",
            "cellular dynamics",
            "molecular kinetics",
            "cellular kinetics",
            "Schur decomposition",
            "Schur vectors",
        }
    ),
)
