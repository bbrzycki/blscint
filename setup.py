import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='blscint',
    version='0.0.1',
    author='Bryan Brzycki',
    author_email='bbrzycki@berkeley.edu',
    description='SETI scintillation utilities',
    long_description=long_description,
    long_description_content_type='text/markdown',
#     url='https://github.com/bbrzycki/setigen',
#     project_urls={
#         'Documentation': 'https://setigen.readthedocs.io/en/latest/',
#         'Source': 'https://github.com/bbrzycki/setigen'
#     },
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
       'numpy>=1.18.1',
       'scipy>=1.4.1',
       'astropy>=4.0',
       'blimpy>=2.0.0',
       'setigen>=2.0.6',
       'galpy>=1.7.2',
       'matplotlib>=3.1.3',
       'seaborn>=0.11.2',
       'pandas>=1.3.5',
       'tqdm>=4.47.0',
       'sphinx-rtd-theme>=0.4.3',
       'sphinx-theme==1.0',
    ],
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ),
)
