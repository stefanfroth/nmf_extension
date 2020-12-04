from setuptools import setup, find_packages
import os

def open_file(fname):
    return open(os.path.join(os.path.dirname(__file__), fname))

setup(
    name='nmf_extension',
    version='0.0.1',
    author='Stefan Roth',
    author_email='stefan@spiced-academy.com',
    packages=find_packages(),
    url='https://github.com/stefanfroth/nmf_extension',
    license="MIT",
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
    description='NMF with missing data',
    long_description=open_file('README.md').read(),
    # dependencies for your library
    install_requires=[
        'scikit-learn'
    ]
)
