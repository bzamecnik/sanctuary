from setuptools import setup, find_packages

setup(name='sanctuary',
      version='0.1.2',
      description='Managed machine-learning model training tool based on sacred.',
      url='https://github.com/bzamecnik/sanctuary',
      author='Bohumir Zamecnik',
      author_email='bohumir.zamecnik@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['tests']),
      zip_safe=False,
      install_requires=[
        'keras>=1.1.1',
        'numpy',
        'sacred',
      ],
      setup_requires=['setuptools-markdown'],
      long_description_markdown_filename='README.md',
      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',

          'License :: OSI Approved :: MIT License',

          'Programming Language :: Python :: 3',

          'Operating System :: POSIX :: Linux',
          'Operating System :: MacOS :: MacOS X',
          ]
      )
