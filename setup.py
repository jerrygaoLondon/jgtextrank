#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.command.install import install as _install

def readme():
    with open('readme.txt') as f:
        return f.read()


class InstallNLTKData(_install):
    def run(self):
        _install.do_egg_install(self)
        import nltk
        nltk.download("punkt")
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')

setup(
    name='jgtextrank',

    # Versions should comply with PEP440.
    # see https://packaging.python.org/en/latest/single_source_version.html
    version='0.1.3',

    description='Yet another Python implementation of TextRank: package for the creation, manipulation, '
                'and study of TextRank algorithm based keywords extraction and summarisation',

    long_description=readme(),

    # The project's main homepage.
    url='https://github.com/jerrygaoLondon/jgtextrank',

    # Author details
    author='Jie Gao',
    author_email='j.gao@sheffield.ac.uk',

    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Intended Audience :: Education',
      'Intended Audience :: Information Technology',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT License',
      'Operating System :: OS Independent',
      'Programming Language :: Python :: 3 :: Only',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Scientific/Engineering :: Human Machine Interfaces',
      'Topic :: Scientific/Engineering :: Information Analysis',
      'Topic :: Text Processing :: General',
      'Topic :: Text Processing :: Indexing',
      'Topic :: Text Processing :: Linguistic',
      'Topic :: Text Processing :: Filters',
      'Topic :: Text Processing :: Linguistic',
    ],

    # What does your project relate to?
    keywords='textrank, parsing, natural language processing, nlp, keywords extraction, '
             'term extraction, text summarisation, text analytics, text mining, '
             'feature extraction, machine learning, graph algorithm, computational linguistics',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=[
          'nltk',
          'networkx'
    ],
    setup_requires=['nltk'],
    cmdclass = {'install':InstallNLTKData},

    extras_require={
        'dev': ['check-manifest', 'matplotlib'],
        'test': ['coverage', 'matplotlib', 'scipy'],
    },

    zip_safe=False)