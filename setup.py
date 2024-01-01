from setuptools import setup, find_packages

setup(name="diffusion",
      version="0.1",
      description="Diffusion using jax and equinox",
      author="Augusto Peres",
      packages=find_packages(),
      install_requires=[
          "jax",
          "equinox",
          "optax",
          "jaxlib",
      ])
