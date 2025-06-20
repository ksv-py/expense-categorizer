from setuptools import find_packages, setup
from typing import List

HYPHER_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    '''
    This functions reads a requirements file 'requirements.txt' and returns a list of requirements.
    It removes any line containing '-e .' which is used for editable installation.

    Args:
        file_path (str): The path to the requirements file.

    Returns:
        List[str]: A list of package requirements.
    '''

    requiremts = []
    with open(file_path, 'r') as file_obj:
        lines = file_obj.readlines()
        requiremts = [req.strip() for req in lines]

    if HYPHER_E_DOT in requiremts:
        requiremts.remove(HYPHER_E_DOT)
    
    return requiremts

setup(
    name='expense-categorizer',
    version='0.0.1',
    author='Keshav Jangid',
    author_email='keshavjangid301@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')

)