from setuptools import setup, find_packages

setup(
    name='promptvista',
    version='0.1.0',
    description='Multi-model prompt experimentation toolkit',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'torch',
        'openai',
        'streamlit',
        'requests',
        'pytest',
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'promptvista=promptvista.ui:launch_ui',
        ],
    },
)
