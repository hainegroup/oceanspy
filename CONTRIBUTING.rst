.. _contributing:

============
Contributing
============
| Report bugs and submit feedback at https://github.com/malmans2/oceanspy/issues.
| Don't forget to add yourself to the list of :ref:`people` contributing to OceanSpy! 

.. _using_git:

Using Git and GitHub
--------------------

Git_ is the distributed version control system used to develop OceanSpy, while GitHub_ is the website hosting the ``oceanspy/`` repository.

**Go to** GitHub_:

1. If you don't have an account yet, Sign up. Otherwise, Sign in. 

2. Go to the OceanSpy_ GitHub repository, then fork the project using the fork button.

**Move to your terminal**:

3. Set your GitHub username and email address using the following commands:

   .. code-block:: bash

    $ git config --global user.email "you@example.com"
    $ git config --global user.name "Your Name"

4. Create a local clone:

   .. code-block:: bash 

    $ git clone https://github.com/your_username_here/oceanspy.git

5. Move into your local clone directory, then set up a remote that points to the original:

   .. code-block:: bash
    
    $ cd oceanspy
    $ git remote add upstream https://github.com/malmans2/oceanspy.git

6. Make a new branch from ``upstream/master``:

   .. code-block:: bash
        
    $ git fetch upstream
    $ git checkout -b name_of_your_new_branch

7. Make sure that your new branch is up-to-date:

   .. code-block:: bash

    $ git merge upstream/master

8. Edit and/or add new files:

    * :ref:`documentation`
    * :ref:`code`

9. To stage files ready for a commit, use the following command:

   .. code-block:: bash
           
    $ git add .

10. To save changes, use the following command:

   .. code-block:: bash 
               
    $ git commit -m "Message describing your edits" 

   You can repeat ``git add`` and ``git commit`` multiple times before pushing the branch online.

11. To push the branch online, use the following command:

   .. code-block:: bash
           
    $ git push -u origin name_of_your_branch

12. Go to your OceanSpy fork on GitHub_ *(https://github.com/your_username_here/oceanspy)* and click on ``Compare and Pull``.

13. Finally, click on ``Send pull request`` button to finish creating the pull request.

.. _documentation:

Contributing to the Documentation
---------------------------------

Documentation link: |docs|

The documentation is built with Sphinx_ and hosted by `Read the Docs`_.
It is written in reStructuredText_.

1. First, you need a local clone of ``oceanspy`` and a branch (follow the instruction in :ref:`using_git`).

2. Move into the directory containing the documentation:

   .. code-block:: bash 
           
    $ cd oceanspy/docs

3. In order to build the documentation, you need to create a Conda_ environment:

   .. code-block:: bash 

    $ conda config --set channel_priority strict
    $ conda config --prepend channels conda-forge
    $ conda env create -f environment.yml

4. Activate the ``ospy_docs`` environment:

   .. code-block:: bash

    $ conda activate ospy_docs

4. Edit and/or add new files.

5. To build the documentation, use the following command:

   .. code-block:: bash
           
    $ make html

   If you want to start from a clean build, run ``make clean`` before ``make html``.

6. You can find the HTML output in ``oceanspy/docs/_build/html``.

7. Use git to ``add``, ``commit``, and ``push`` as explained in :ref:`using_git`.


.. _code:

Contributing to the Code
------------------------

Continuous Integration and Test Coverage links: |travis| |codecov|

1. First, you need a local clone of ``oceanspy`` and a branch (follow the instructions in :ref:`using_git`).

2. If you are not already into your local clone directory, move there:

   .. code-block:: bash
           
    $ cd oceanspy

3. Create a test environment (substituting ``pyxx`` with one of the available python versions):

   .. code-block:: bash
    
    $ conda config --set channel_priority strict
    $ conda config --prepend channels conda-forge
    $ conda env create -f ci/environment-pyxx.yml

4. Activate the test environment:

   .. code-block:: bash

    $ conda activate ospy_tests

5. Install OceanSpy in development mode:

   .. code-block:: bash 
           
    $ pip install -e .

6. Edit and/or add new files.

7. Use git to ``add``, ``commit``, and ``push`` as explained in :ref:`using_git`.

8. Make sure that the code is well tested by adding or improving tests in the ``oceanspy/tests`` repository. The python package used to test OceanSpy is pytest_. Use the following command to run the test and measure the code coverage:

   .. code-block:: bash 

    $ py.test oceanspy -v --cov=oceanspy --cov-config .coveragerc --cov-report term-missing

9. You can install and use `pytest-html`_ to produce a test report in html format.

10. Make sure that the code follows the `PEP 8`_ style guide using the following command:

   .. code-block:: bash 
           
    $ pycodestyle oceanspy


Deploying
---------

Python Package Index (PyPI) link: |version|

A reminder for the maintainers on how to deploy.

1. Download and install bump2version

   .. code-block:: bash

    $ pip install --upgrade bump2version

2. Update ``HISTORY.rst``

3. Use git to ``add`` and ``commit`` changes

4. Update version number

   .. code-block:: bash

    $ bump2version patch # possible: major / minor / patch

5. Release on PyPI_ by uploading both sdist and wheel:

   .. code-block:: bash

    $ python setup.py sdist upload
    $ python setup.py bdist_wheel upload 

6. Use git to ``push``

7. Push tags

   .. code-block:: bash

    $ git push --tags

8. Add the release's notes on the `releases' page`_ (copy and paste from ``HISTORY.rst``)
   

.. _Git: https://git-scm.com
.. _GitHub: https://github.com
.. _OceanSpy: https://github.com/malmans2/oceanspy
.. _Sphinx: http://www.sphinx-doc.org/en/master
.. _`Read the Docs`: https://readthedocs.org
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _Conda: https://conda.io/docs
.. _PyPI: https://pypi.org/project/oceanspy
.. _`releases' page`: https://github.com/malmans2/oceanspy/releases
.. _pytest: https://docs.pytest.org/en/latest
.. _`pytest-html`: https://pypi.org/project/pytest-html
.. _`PEP 8`: https://www.python.org/dev/peps/pep-0008


.. |docs| image:: http://readthedocs.org/projects/oceanspy/badge/?version=latest
    :alt: Documentation
    :target: http://oceanspy.readthedocs.io/en/latest/?badge=latest

.. |travis| image:: https://travis-ci.org/malmans2/oceanspy.svg?branch=master
    :alt: Travis
    :target: https://travis-ci.org/malmans2/oceanspy
    
.. |codecov| image:: https://codecov.io/github/malmans2/oceanspy/coverage.svg?branch=master
    :alt: Coverage
    :target: https://codecov.io/github/malmans2/oceanspy?branch=master

.. |version| image:: https://img.shields.io/pypi/v/oceanspy.svg?style=flat
    :alt: PyPI
    :target: https://pypi.python.org/pypi/oceanspy


