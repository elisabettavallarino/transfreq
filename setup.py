from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='transfreq', # Package name
    version='1.1',
    description='Python package to compute the transition frequency from theta to alpha band from MEEG data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url='<optional>',
    author='Elisabetta Vallarino, Sara Sommariva',
    author_email='vallarino@dima.unige.it',
    # license='<optional>',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython'
    ],
    # keywords='<optional>',
    packages=find_packages(exclude=['docs', 'tests']),
    # setuptools > 38.6.0 needed for markdown README.md
    setup_requires=['setuptools>=38.6.0'],
    include_package_data=True,
    package_data={'': ['data/*.fif']}
)
