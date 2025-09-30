from setuptools import setup, find_packages

# Helper to load requirements from requirements.txt
def load_requirements(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Load the README for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qcm",
    version="0.1.0",
    author="Bryan Cardenas",
    description="Exploring Quantum Latent for Medicine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="<your-github-repo-url>", # Add your project's URL here
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=load_requirements("requirements.txt"),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Or another license of your choice
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
