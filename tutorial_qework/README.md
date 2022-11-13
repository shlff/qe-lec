# Some tutorials on QuantEcon working process

(Matt's advice: put more contexts on the work flow.)


## 1 general working process

- step 1: ``git checkout master``
- step 2: ``git pull``
- step 3: open, edit and save file(s) 
- step 4: ``git add -A``
- step 5: ``git status``
- step 6: ``git commit -m "xxx"``
- step 7: ``git push``
- step 8: ``git checkout master``

## 2 squash commits

- step 1: ``git checkout master``
- step 2: ``git pull``
- step 3: ``git checkout <branch_name>``
- step 4: ``git pull``
- step 5: open, edit and save file(s)
- step 6: ``git add -A``
- step 7: ``git status``
- step 8: ``git commit -m "<message>"``
- step 9: ``git reset --soft HEAD~#``
  - ``#`` = # of commits in pull request + # of commits created from ``step 8``
- step 10: ``git commit -m "<new_message>"``
- step 11: ``git push origin +<branch_name>``
- step 12: ``git checkout master``

## 3 check and reset commit labels

- step 1: ``git checkout <branch_name>``
- step 2: ``git log``
- step 3: ``git reset --hard <the_commit_previous_to_the_one_we_want_to_reset>``
- step 4: ``git push --force``

## 4 keep current with the master branch

- step 1: ``git checkout master``
- step 2: ``git pull``
- step 3: ``git checkout <my-feature-branch>``
- step 4: ``git push origin <my-feature-branch>``
- step 5: ``git rebase master``
- step 6: ``git push origin <my-feature-branch> --force``

For inference, please see [here](https://gist.github.com/blackfalcon/8428401#keeping-current-with-the-master-branch).

## 5 setups for texlive

reference:
- https://tug.org/mactex/mactex-unix-download.html

1. Download [install-tl-unx.tar.gz](https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz) to your current directory.
2. Run the following code in the terminal:
    ```
    zsh
	umask 0022
	sudo perl install-tl -gui text
    ```
3. Locate to the to-be-compiled file, ``xx.tex``, and run
   ```
   xelatex -shell-escape xx.tex
   ```
4. Then run
   ```
   bibtex xx
   ```
   to locate the bib file
5. Run
   ```
   xelatex -shell-escape xx.tex
   ```
   again

## 6 adding commits to a fork

To do that, we need to add a remote user repo in our local machine.

For example:
1. git remote add crondonm https://github.com/crondonm/QuantEcon.py.git
2. git fetch crondonm
3. git checkout -t crondonm/VAR-approximation

Now we will be in the branch VAR-approximation of crondonm and we can add our changes/commits and just push.
