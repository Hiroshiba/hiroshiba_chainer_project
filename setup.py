from setuptools import find_packages, setup

setup(
    name='project_name',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/Hiroshiba/project_name',
    author='Kazuyuki Hiroshiba',
    author_email='hihokaruta@gmail.com',
    install_requires=[
        'chainer',
    ],
)
