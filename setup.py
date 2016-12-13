import os
import subprocess
import sys
import contextlib

import numpy
import murmurhash
from setuptools import Extension, setup


VENDOR = 'texture'


PACKAGE_MODS = [
        'texture.count_matrix',
]


PACKAGES = ['texture']


@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


def clean(path):
    for name in PACKAGE_MODS:
        name = name.replace('.', '/')
        for ext in ['.c', '.cpp', '.so', '.html']:
            file_path = os.path.join(path, name + ext)
            if os.path.exists(file_path):
                os.unlink(file_path)


def get_python_package(root):
    return os.path.join(root, VENDOR)


def generate_sources(root):
    for base, _, files in os.walk(root):
        for filename in files:
            if filename.endswith('pyx'):
                yield os.path.join(base, filename)


def cythonize_sources(root):
    print("Cythonizing sources")
    source = get_python_package(root)
    print("Processing %s" % source)

    try:
        p = subprocess.call(['./bin/cythonize.py'] + [source])
        if p != 0:
            raise Exception('Cython failed')
    except OSError:
        raise OSError('Cython needs to be installed')


def generate_extensions(root, macros=[]):
    ext_modules = []
    for mod_name in PACKAGE_MODS:
        mod_path = mod_name.replace('.', '/') + '.cpp'
        ext_modules.append(
            Extension(mod_name,
                      sources=[mod_path],
                      extra_compile_args=['-O3', '-fPIC'],
                      include_dirs=[murmurhash.get_include(), numpy.get_include()],
                      define_macros=macros))

    return ext_modules


def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))

    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        return clean(root)

    cython_cov = 'CYTHON_COV' in os.environ

    macros = []
    if cython_cov:
        print("Adding coverage information to cythonized files.")
        macros =  [('CYTHON_TRACE', 1)]

    with chdir(root):
        cythonize_sources(root)
        ext_modules = generate_extensions(root, macros=macros)
        setup(
            name='texture',
            version='0.1.0',
            description='Text Feature Generation Using SpaCy',
            author='Joshua D. Loyal',
            url='https://github.com/joshloyal/Texture',
            license='MIT',
            install_requires=['numpy', 'spacy'],
            packages=PACKAGES,
            ext_modules=ext_modules,
        )


if __name__ == '__main__':
    setup_package()
