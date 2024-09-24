'''
Its objective is to ensure that program is installed correctly, 
...using ‘pip’ use setup.py to install any module wo calling setup.py directly
Helps in creating ML application as a package, 
.. helps in deployment,which can be installed by others or the user itself 

'''
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT='-e .'
def get_requirements(path:str)->List[str]:
    '''
    returns a list of requirements
    '''
    requirements=[]
    with open(path) as f:
        requirements=f.readlines()
        requirements=(req.replace("\n", "")
                      for req in requirements)
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

# project metadata
setup(
    name='Project_One',
    version='0.0.1',
    author='Anurag',
    author_email='anurag9996211@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)
