from setuptools import setup

setup(
    name='pse',
    version='1.0',
    packages=['pse'],
    url='',
    license='MIT',
    author='Frank Heinrich',
    author_email='fheinrich@cmu.edu',
    description='GP and Gridsearch Phase Space Exploration',
    install_requires=[
        'aio-pika>=9.0',
        'roadmap-broker-client',
    ],
)
