from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    '''
    This functions reads a requirements file 'requirements.txt' and returns a list of requirements.
    It removes any line containing '-e .' which is used for editable installation.

    Args:
        file_path (str): The path to the requirements file.

    Returns:
        List[str]: A list of package requirements.
    '''

    requirements = []
    with open(file_path, 'r',encoding="utf-8", errors="replace") as file_obj:
        lines = file_obj.readlines()
        requirements = [req.strip() for req in lines if req.strip() and req.strip() != HYPHEN_E_DOT]
    return requirements

setup(
    name='expense-categorizer',
    version='0.0.1',
    author='Keshav Jangid',
    author_email='keshavjangid301@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=get_requirements('requirements.txt')
)