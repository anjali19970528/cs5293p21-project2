from setuptools import setup, find_packages

setup(
    name='unredactor',
    version='1.0',
    author='You Name',
    author_email='your ou email',
    packages=find_packages(exclude=('tests', 'docs')),
    setup_requires=['pytest-runner'],
    tests_require=['pytest']	
)

