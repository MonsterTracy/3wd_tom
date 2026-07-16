from setuptools import find_packages, setup


setup(
    name="werewolf-tom",
    version="1.0.0",
    description="First- and second-order wolf-pair belief modeling",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24,<3.0",
        "openai>=1.59.3,<3.0",
        "python-dotenv>=1.0,<2.0",
        "PyYAML>=6.0,<7.0",
        "torch>=2.0,<3.0",
    ],
    extras_require={"dev": ["pytest>=8.0,<10.0"]},
)
