from setuptools import setup, find_packages

setup(
    name="fmz_dataset_processor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "tqdm",
        "chardet",
        "python-magic",  # 用于检测文件编码
    ],
    entry_points={
        'console_scripts': [
            'process_fmz_dataset=process_fmz_dataset:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool to process FMZ strategy datasets",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fmz_dataset_processor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
