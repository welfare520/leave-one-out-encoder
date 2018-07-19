#!/usr/bin/env python

from setuptools import setup

setup(name='loo_encoder',
      version='0.0.3',
      description='Leave one out encoding of categorical features',
      url='https://github.com/welfare520/leave-one-out-encoder',
      author='He Zhang',
      author_email='zhanghe.dr@gmail.com',
      license='GPL v3',
      packages=['loo_encoder'],
      install_requires=[
          'pandas', 'numpy'
      ],
      zip_safe=False)
