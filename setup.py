import os 
import sys
from setuptools import setup, find_packages

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> list:
    'this function get thr list of requirements'
    requirements=[]
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirement= [req for req in requirements if req!= HYPHEN_E_DOT or req!= '\n']
    return requirement

setup(
    name ='Urban Scene Object Detection',
    version='0.0.1',
    author='faker_112',
    author_email='',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)