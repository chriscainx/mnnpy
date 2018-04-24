from setuptools import setup

setup(name='mnnpy',
      version='0.1',
      description='Yet another python package.',
      url='http://github.com/chriscainx/mnnpy',
      author='Chris Kang',
      author_email='kbxchrs@gmail.com',
      license='BSD 3',
      packages=['mnnpy'],
      install_requires=[
          'numpy',
          'anndata',
          'scipy',
          'pandas',
          'numba'
      ],
      zip_safe=False)
