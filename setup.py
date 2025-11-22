import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="micrograd",
    version="0.0.0",
    author="Vatsal Patel",
    description="A tiny scalar-valued autograd engine with a small PyTorch-like neural network library on top. Copied from Andrej Karpathy's micrograd.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vatsalmpatel/Micrograd-From-Scratch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)