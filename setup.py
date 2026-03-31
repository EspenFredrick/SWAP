from setuptools import setup, find_packages

setup(
    name="swap",
    version="0.1.0",
    description="Solar Wind Analysis and Propagation toolkit",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "spacepy",
        "pyspedas",
        "python-dateutil",
    ],
)
