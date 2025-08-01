"""
fMRI-MAE: Masked Autoencoder for fMRI Data Analysis

A PyTorch implementation of Masked Autoencoders for self-supervised learning 
on functional Magnetic Resonance Imaging (fMRI) data.
"""

from setuptools import setup, find_packages
import os

# Read version
def read_version():
    version_file = os.path.join(os.path.dirname(__file__), 'src', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read README
def read_readme():
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_file):
        with open(requirements_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="fmri-mae",
    version=read_version(),
    author="Research Team",
    author_email="research@example.com",
    description="Masked Autoencoder for fMRI Data Analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/fmri-mae",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "ipywidgets>=7.6.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
            "bokeh>=2.4.0",
        ]
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "fmri-mae-train=scripts.train:main",
            "fmri-mae-evaluate=scripts.evaluate:main",
        ],
    },
    
    # Package data
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "*.md", "*.txt"],
    },
    
    # Metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    
    # Keywords for discovery
    keywords=[
        "fmri", "neuroimaging", "masked-autoencoder", "self-supervised-learning",
        "pytorch", "brain-connectivity", "medical-ai"
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/username/fmri-mae/issues",
        "Source": "https://github.com/username/fmri-mae",
        "Documentation": "https://fmri-mae.readthedocs.io/",
    },
)
