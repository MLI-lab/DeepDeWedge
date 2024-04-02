from os import path

from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ddw',
    setup_requires=['setuptools_scm'],
    python_requires='>=3.7.0',
    author='Simon Wiedemann',
    author_email='simonw.wiedemann@tum.de',
    description='Simultaneous denoising and missing wedge reconstruction of cryo-ET tomograms.',
    packages=['ddw', 'ddw.utils'],
    entry_points={
        'console_scripts': [
            'ddw = ddw.app:main',
        ]},
)