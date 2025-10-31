import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name='population-error',
    version='0.1.2',
    include_package_data=False,
    description='JAX-based package for estimating the information lost due to Monte Carlo approximations in GW population inference.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jack-heinzel/population-error',
    project_urls={
        'Source': 'https://github.com/jack-heinzel/population-error',
        'Documentation': 'https://population-error.readthedocs.io/',
    },
    author='Jack Heinzel',
    install_requires=['jax', 'jax_tqdm', 'gwpopulation', 'bilby'],
    author_email='heinzelj@mit.edu',
    packages=["population_error"],
    zip_safe=False
)
