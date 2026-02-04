from setuptools import setup, find_packages

setup(
    name="alertdrive",
    version="0.1.0",
    description="Driver Drowsiness Detection - Comparative Analysis",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "opencv-python>=4.8.0",
        "tensorflow>=2.13.0",
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
)
