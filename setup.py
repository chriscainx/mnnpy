from setuptools import setup

setup(name='mnnpy',
      version='0.1.0',
      description='Mutual nearest neighbors correction in python.',
      long_description='Correcting batch effects in single-cell expression datasets using the mutual nearest neighbors method.',
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
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],
      python_requires='>=3.4',
      py_modules=['irlb', 'mnn', 'utils'],
      zip_safe=False)
