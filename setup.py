#!/usr/bin/env python

from setuptools import setup

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='loo_encoder',
    version='0.0.8',
    description='Leave one out encoding of categorical features',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/welfare520/leave-one-out-encoder',
    author='He Zhang',
    author_email='zhanghe.dr@gmail.com',
    license='GPL v3',
    packages=['loo_encoder'],
    install_requires=[
        'pandas', 'numpy'
    ],
    zip_safe=False
)
