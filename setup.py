import setuptools
import os

_PROJECT_AUTHOR = "Omdeep Borkar"
_PROJECT_AUTHOR_EMAIL = "omdeeborkar@gmail.com"
_PROJECT_URL = "https://github.com/Omdeepb69/Kinectic-Command"
_PROJECT_NAME = "kinectic_command"
_PROJECT_VERSION = "0.1.0"
_PROJECT_DESCRIPTION = (
    "An AI system leveraging real-time computer vision to interpret hand gestures "
    "from a webcam, translating them into control commands for a simulated drone "
    "or robotic arm. It explores intuitive, 'minority report'-style non-contact "
    "human-machine interfaces."
)

def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return _PROJECT_DESCRIPTION # Fallback to short description if README not found

setuptools.setup(
    name=_PROJECT_NAME,
    version=_PROJECT_VERSION,
    author=_PROJECT_AUTHOR,
    author_email=_PROJECT_AUTHOR_EMAIL,
    description=_PROJECT_DESCRIPTION,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url=_PROJECT_URL,
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python>=4.5",
        "mediapipe>=0.8",
        "numpy>=1.19",
        "pygame>=2.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8',
    project_urls={
        "Bug Tracker": f"{_PROJECT_URL}/issues",
        "Source Code": _PROJECT_URL,
    },
    keywords='computer vision, hand tracking, gesture recognition, human-computer interaction, mediapipe, opencv, pygame, drone control, robotics',
)