from setuptools import find_packages,setup
from typing import List



error="-e ."
def get_requirements(file_path:[str])->List[str]:
    requirements=[]
    with open (file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("/n"," ") for req in requirements]


        if error in requirements:
            requirements.remove(error)
    return requirements


setup(
name="machine project",
version="0.0.1",
author="priyanshu",
email="priyanshuc111@gmail.com",
setup=find_packages(),
install_requires=get_requirements("requirements.txt")

)




