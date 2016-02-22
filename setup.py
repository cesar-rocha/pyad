from setuptools import setup, Extension

VERSION='0.1d'

DISTNAME='pyad'
URL='https://github.com/crocha700/pyad'
# how can we make download_url automatically get the right version?
DOWNLOAD_URL='https://github.com/crocha700/pyad/v%s' % VERSION
AUTHOR='Cesar B Rocha'
AUTHOR_EMAIL='crocha@ucsd.edu'
LICENSE='MIT'

DESCRIPTION='python spectral model of advection-diffusion'
LONG_DESCRIPTION="""
"""

CLASSIFIERS = [
    'Development Status :: 1 - Beta',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Scientific/Engineering :: Atmospheric Science'
]


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pylattice',
      version='0.1d',
      description='Pythonian model of advection=diffusion\
              analysis',
      url='http://github.com/crocha700/pylattice',
      author='Cesar B Rocha',
      author_email='crocha@ucsd.edu',
      license='MIT',
      packages=['pyad'],
      install_requires=[
          'numpy',
      ],
      test_suite = 'nose.collector',
      zip_safe=False)
