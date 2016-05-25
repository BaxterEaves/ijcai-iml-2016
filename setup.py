from setuptools import setup
from setuptools.command.test import test as TestCommand
from distutils.core import Extension
from pip.req import parse_requirements
from Cython.Build import cythonize

import sys
import os

DIR = os.path.dirname(os.path.realpath(__file__))
TEST_DIR = os.path.join(DIR, 'ldateach', 'tests')


class UnitTest(TestCommand):
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        errno = pytest.main([TEST_DIR])
        sys.exit(errno)


extensions = [Extension("ldateach.fastlda",
                        sources=[os.path.join('ldateach', 'fastlda.pyx'),
                                 os.path.join('src', 'lda.cpp'),
                                 os.path.join('src', 'utils.cpp'),
                                 os.path.join('src', 'dist.cpp'),
                                 ],
                        extra_compile_args=['-std=c++11', '-Wall'],
                        undef_macros=["NDEBUG"],
                        include_dirs=['src'],
                        language="c++",)]

extensions = cythonize(extensions)

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='ldateach',
    version='0.1.0-dev',
    author='Baxter S. Eaves Jr.',
    url='https://github.com/CoDaS-Lab/teaching-topics',
    long_description='Code used for IJCAI 2016 paper.',
    install_requires=reqs,
    tests_require=['pytest'],
    package_dir={'ldateach': 'ldateach/'},
    cmdclass={'test': UnitTest},
    ext_modules=extensions,
)
