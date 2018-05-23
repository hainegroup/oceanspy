.. highlight:: shell

============
Contributing
============

Report Bugs and Submit Feedbacks
--------------------------------

Report bugs and submit feedbacks at https://github.com/malmans2/oceanspy/issues.




.. _git:

Using Git and GitHub
--------------------

Git is the distributed version control system used to develop OceanSpy, while GitHub_ is the website hosting the repository ``oceanspy``.

**Go to** GitHub_:

1. If you don't have an account yet, Sign up. Otherwise, Sign in. 

2. Go to the OceanSpy_ GitHub repository, then fork the project using the fork button (top right).

**Move to your terminal**:

3. Set your GitHub username and email address. From your terminal (e.g., SciServer's terminal), run the following commands

   .. code-block:: bash

    git config --global user.name "your_username_here"
    git config --global user.email your_email_here@example.com

4. Create a local clone

   .. code-block:: bash 

    git https://github.com/your_username_here/oceanspy.git

5. Move into your local clone directory, then set up a remote that points to the original

   .. code-block:: bash
    
    cd oceanspy
    git https://github.com/malmans2/oceanspy.git

6. Make a new branch from ``upstream/master``

   .. code-block:: bash
        
    git fetch upstream
    git checkout -b name_of_your_new_branch

7. Edit and/or add new files

    * :ref:`documentation`
    * :ref:`code`

8. To stage the files ready for a commit, use the command

   .. code-block:: bash
           
    git add .

9. To commit the files, use the command

   .. code-block:: bash 
               
    git commit -m "Commit message describing your edits" 

10. You can use multiple commits, and repeat 8 and 9 multiple times.
11. To push your branch and update your GitHub copy of ``oceanspy``, use the command

   .. code-block:: bash
           
    git push -u origin name_of_your_branch

**Finally, go to your OceanSpy fork on** GitHub_ *(https://github.com/your_username_here/oceanspy)* **and click on** ``Compare and Pull``.
   




.. _documentation:

Contributing to the Documentation
---------------------------------
The documentation is built with Sphinx_ and hosted by `Read the Docs`_.
It is written in reStructuredText_.

1. First, you need a local clone of ``oceanspy`` and a branch (follow the instruction in :ref:`git`)

2. Move into the directory containing the documentation

   .. code-block:: bash 
           
    cd oceanspy/docs

3. In order to build the documentation, you need to create a Conda_ environment

   .. code-block:: bash 
           
    conda env create -f environment.yml

4. Activate the new environment (named ``rtd``)
   
   .. code-block:: bash

    # Older versions of conda
    source activate rtd 
    # Newer versions of conda
    conda activate rtd

   NOTE: make sure that the ``rtd`` environment is activated every time you work on the documentation.
   For example, if you start working on a new terminal, you don't need to create a new environment but you just have to activate ``rtd``.

4. Edit and/or add new files

5. To build the documentation run:

   .. code-block:: bash
           
    make html

   NOTE: if you want to do a full clean build, run ``make clean`` before ``make html``.

6. You can find the HTML output in ``ocenspy/docs/_build/html``.

7. Use git to ``add``, ``commit``, and ``push`` as explained in :ref:`git`.






.. _code:

Contributing to the Code
------------------------

1. First, you need a local clone of ``oceanspy`` and a branch (follow the instructions in :ref:`git`)

2. If you are not already into your local clone directory, move there

   .. code-block:: bash
           
    cd oceanspy

3. Install OceanSpy's dependencies, following the instruction in :ref:`dependencies`.

   TODO: no need to create a separate environment yet.
   In the future, use ``conda env create -f ci/environment-pyxx.yml`` and add tests! 

4. Install OceanSpy in development mode

   .. code-block:: bash 
           
    pip install -e .

5. Edit and/or add new files

6. Use git to ``add``, ``commit``, and ``push`` as explained in :ref:`git`.





Deploying
---------

A reminder for the maintainers on how to deploy.

1. Download and install bumpversion

   .. code-block:: bash

    pip install --upgrade bumpversion

2. Update ``HISTORY.rst``

3. Use git to ``add`` and ``commit`` changes

4. Update version number

    .. code-block:: bash

    bumpversion patch # possible: major / minor / patch

5. Release on PyPI_ by uploading both sdist and wheel:

    .. code-block:: bash

    python setup.py sdist upload
    python setup.py bdist_wheel upload 

6. Use git to ``push``

7. Push tags

    .. code-block:: bash

    git push --tags

8. Add the release's notes on the `releases' page`_ (copy and past from ``HISTORY.rst``)
   

.. _GitHub: https://github.com
.. _OceanSpy: https://github.com/malmans2/oceanspy
.. _Sphinx: http://www.sphinx-doc.org/en/master
.. _Read the Docs: https://readthedocs.org
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _Conda: https://conda.io/docs
.. _PyPI: https://pypi.org/project/oceanspy
.. _releases' page: https://github.com/malmans2/oceanspy/releases


