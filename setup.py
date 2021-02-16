from setuptools import setup
from setuptools import find_packages

setup(name='la-gcn-torch',
      version='0.1',
      description='Learnable Aggregator for Graph Convolutional Networks in PyTorch',
      author='Ahmet Sarıgün',
      author_email='asarigun3156@gmail.com',
      url='https://github.com/asarigun',
      download_url='https://github.com/asarigun/la-gcn-torch',
      license='MIT',
      install_requires=['numpy',
                        'torch',
                        'scipy'
                        ],
      package_data={'la-gcn-torch': ['README.md']},
      packages=find_packages())
