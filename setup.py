from setuptools import setup
import versioneer

NAME = 'dsphsim'
CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
Programming Language :: Python
Natural Language :: English
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Physics
Topic :: Scientific/Engineering :: Astronomy
Operating System :: MacOS
Operating System :: POSIX
License :: OSI Approved :: MIT License
"""
URL = 'https://github.com/kadrlica/%s'%NAME
DESC = "Simulate dwarf galaxy stellar distributions"
LONG_DESC = "See %s"%URL

setup(
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url=URL,
    author='Alex Drlica-Wagner',
    author_email='kadrlica@fnal.gov',
    scripts = ['bin/dsphsim'],
    install_requires=[
        'python >= 2.7.0',
        'ugali >= 1.6.0',
        'vegas >= 3.0',
    ],
    packages=['dsphsim'],
    package_data={'dsphsim':['data/*.dat']},
    description=DESC,
    long_description=LONG_DESC,
    platforms='any',
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f]
)
