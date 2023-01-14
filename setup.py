from setuptools import setup, find_packages
import shutil
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

if os.path.exists("./build"):
    shutil.rmtree("./build")
if os.path.exists("./dist"):
    shutil.rmtree("./dist")
if os.path.exists("./pure_ldp.egg-info"):
    shutil.rmtree("./pure_ldp.egg-info")

setup(
    name='pure-ldp',
    version='1.1.2',
    packages=find_packages(exclude=['*development*',
                                    "*hashtogram*", "*explicit_hist*", ]),

    install_requires=["xxhash", "numpy", "scipy", "bitstring", "bitarray"],
    url='https://github.com/Samuel-Maddock/pure-LDP',
    license='MIT',
    author='Samuel Maddock',
    author_email='samuel-maddock@hotmail.com',
    description='Simple pure LDP frequency oracle implementations',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
