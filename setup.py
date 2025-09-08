import setuptools

setuptools.setup(
    name='population-error',
    version='0.1',
    description='JAX-based package for estimating the information lost due to Monte Carlo approximations in GW population inference.',
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
