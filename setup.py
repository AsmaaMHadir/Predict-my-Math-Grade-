from setuptools import find_packages,setup # will find all of the available packages
from typing import List

HYPE_E_DOT = "-e ."

def get_requirements(file_path:str) -> List[str]:
    '''
    retursn the list of requirements
    '''
    requirements =[]
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if HYPE_E_DOT in requirements:
            requirements.remove(HYPE_E_DOT
                                )
            
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Asmaa Hadir',
    author_email="asmaahadir11@gmail.com",
    packages= find_packages(),
    install_requires=get_requirements('requirements.txt'),
)