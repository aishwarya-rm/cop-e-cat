from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_desc = (here / 'README.md').read_text(encoding='utf-8')

setup(
    packages=find_packages(),
    scripts=[],
    long_description=long_desc,
    long_description_content_type='text/markdown',
    install_requires=[
        'pandas>=1.0.4',
        'numpy>=1.20.1',
        'psycopg2>=2.8.5',
        'pytest>=6.2.3'
    ],
    python_requires='>=3'
)