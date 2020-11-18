from setuptools import setup, find_packages

__version__ = "0.0.0"
exec(open('tomography/_version.py').read())

setup(
    name="tomography",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'gp_extras',
        'scikit-image',
        'matplotlib',
        'PyWavelets',
        'GPyOpt'
    ],
    # metadata
    scripts=['tomography/tomorun.py'],
    author="Gioele La Manno",
    author_email="gioelelamanno@gmail.com",
    description="Functions for RNA-seq tomography",
    license="MIT",
    url="https://github.com/linnarsson-lab/tomography",
)
