import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)


# Installs
setuptools.setup(
    name="prioreye",
    version="1.0.0",
    author="Kyuhwan Yeon",
    author_email="kyuhwanyeon@gmail.com",
    description="PriorEye: Geospatial Visual Priors for End-to-End Autonomous Driving",
    url="https://github.com/kyuhwanyeon/prioreye",
    python_requires=">=3.9",
    packages=setuptools.find_packages(script_folder),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
    ],
    license="apache-2.0",
)
