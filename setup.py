import setuptools

with open("README.md", "r") as f:
    readme = f.read()

setuptools.setup(
    name="abeliantensors",
    version="0.1.0",
    author="Markus Hauru",
    author_email="markus@mhauru.org",
    description="Library for Abelian symmetry preserving tensors.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/mhauru/abeliantensors",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    keywords=["tensor", "tensor networks", "linear algebra"],
    install_requires=["scipy>=1.0.0"],
    extras_require={
        "tests": ["ncon", "pytest", "pytest-randomly", "coverage"],
        "demo": ["ncon", "timeit", "matplotlib", "seaborn"],
        "doc": ["sphinx", "sphinx_rtd_theme", "recommonmark"],
    },
    python_requires=">=3.6",
)
