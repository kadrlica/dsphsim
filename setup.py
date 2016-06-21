import sys
import os
try: from setuptools import setup
except ImportError: from distutils.core import setup
import versioneer

here = os.path.abspath(os.path.dirname(__file__))

def read(filename):
    return open(os.path.join(here,filename)).read()

setup(
    name='dsphsim',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url='https://github.com/kadrlica/dsphsim',
    author='Alex Drlica-Wagner',
    author_email='kadrlica@fnal.gov',
    scripts = ['bin/dsphsim'],
    install_requires=[
        'python >= 2.7.0',
        'vegas >= 3.0',
    ],
    packages=['dsphsim'],
    package_data={'dsphsim':['data/*.dat']}
    description="Simple automated interface to scientific wiki tools.",
    long_description=read('README.md'),
    platforms='any',
    keywords='astronomy',
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
    ]
)
