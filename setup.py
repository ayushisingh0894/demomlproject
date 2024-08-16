from setuptools import find_packages,setup
from typing import List

hyphen_dot='-e.'
def get_requirements(file_path:str)->List[str]:
    requirement=[]
    with open('requirement.txt') as file_obj:
        requirement=file_obj.readlines()
        requirement=[req.replace('/n','') for req in requirement]

        if hyphen_dot in requirement:
            requirement.remove(hyphen_dot)
    return requirement
 
setup(
    name='mlproject',
    version='0.0.1',
    author='Krish',
    author_email='a4ayushi0894@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirement.txt')

)