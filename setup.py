from setuptools import setup

setup(name='sanctuary',
      version='0.1',
      description='Managed machine-learning model training tool based on sacred.',
      url='https://github.com/bzamecnik/sanctuary',
      author='Bohumir Zamecnik',
      author_email='bohumir.zamecnik@gmail.com',
      zip_safe=False,
      py_modules=['sanctuary'],
      install_requires=[
        'keras>=1.1.1',
        'numpy',
        'sacred',
      ],
      setup_requires=['setuptools-markdown'],
      long_description_markdown_filename='README.md',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',

          'License :: Other/Proprietary License',

          'Programming Language :: Python :: 3',

          'Operating System :: POSIX :: Linux',
          'Operating System :: MacOS :: MacOS X',
          ]
      )
