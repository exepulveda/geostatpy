# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='geostatpy',  # Required
    version='0.1.0',  # Required
    description='A Python geostatistic tool for educational purposes',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/exepulveda/geostatpy',  # Optional
    author='Exequiel Sepúlveda Escobedo',  # Optional
    author_email='esepulveda@protonmail.com',  # Optional
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: Free for non-commercial use',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='geostatistics kriging',  # Optional
    package_dir={'': 'src'},  # Optional
    packages=find_packages(where='src'),  # Required
    python_requires='>=3.6, <4',
    install_requires=['setuptools','numpy','scipy'],  # Optional
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/exepulveda/geostatpy/issues',
        'Source': 'https://github.com/exepulveda/geostatpy/',
    },
)