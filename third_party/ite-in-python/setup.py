import setuptools

with open("LICENSE.txt", "r") as fh:
    license = fh.read()

with open("README.md", "r") as fh:
    long_description = fh.read()

short_description = (
   "Information Theoretical Estimators (ITE) in Python"
)

setuptools.setup(
    name="ite",
    version="1.1",
    author="Zoltan Szabo",
    author_email="zoltan.szabo@polytechnique. edu",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/szzoli/ite-in-python",
    packages=["ite"],
    license=license,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
