# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='RDSolver',
    version='0.1.0',
    description='Reaction-Diffusion PDE Solver',
    long_description=readme,
    author='Roozbeh H. Pazuki',
    author_email='rpazuki@gmail.com',
    url='https://github.com/rpazuki/RDSolver',
    license=license
)