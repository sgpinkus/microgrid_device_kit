from setuptools import setup, find_packages
import sys
import os


version = '0.1'
setup(
    name='device_kit',
    version=version,
    description="",
    classifiers=[
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.5',
    ],  # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    keywords='',
    author='Sam Pinkus',
    author_email='sgpinkus@gmail.com',
    license='',
    packages=['device_kit', 'device_kit.parser'],
    package_dir={'device_kit': '.'},
    include_package_data=True,
    zip_safe=False,
    install_requires=[
      'numpy',
      'pandas',
      'scipy',
      'numdifftools',
      'matplotlib',
    ],
)
