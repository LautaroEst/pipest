import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pipest", # Replace with your own username
    version="0.0.2",
    author="Lautaro Estienne",
    author_email="lautaro.est@gmail.com",
    description="Python package for machine learning pipelines excecution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LautaroEst/pipest",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)