# Movie-Score-Predictor-Machine-Learning-4612
This is a project I completed for my Introduction to Machine Learning class at Florida International University

This code was developed in the PyCharm Python IDE. The IDE actually handles the downloading of libraries differently than the instructions on the website of any given library.

For instance, the scikit-learn library's website (https://scikit-learn.org/stable/install.html) explains that you could install using a console command depending on whether you use pip or conda as a packager.

But PyCharm actually has a "Python Interpreter" section in it that allows for a user to look up libraries to install into the IDE.
Running this code on PyCharm IDE is recommended, as I do not know how to install the libraries in any other IDE or code editor.

---------------------------------------------------------------------------------------------------------------------------------------------------------------

In Pycharm, you can just create a new project and add the MovieScorer.py file and the 2 datasets (tweets_test.csv and tweets_training.csv) into a project folder (Not the one with the ".idea" file, the one with the "Lib" and "Scripts" files).

---------------------------------------------------------------------------------------------------------------------------------------------------------------

In Pycharm, you can just create a new project and add the MovieScorer.py file and the 2 datasets (tweets_test.csv and tweets_training.csv) into a project folder (Not the one with the ".idea" file, the one with the "Lib" and "Scripts" files).

After doing so, 

There are imports that do not require any external libraries, these are:

csv
re

These are the external libraries that must be installed:

scikit-learn (https://scikit-learn.org/stable/install.html)
NLTK (https://www.nltk.org/install.html)

IF USING PYCHARM:

Open Project, Go to settings, go to project drop down, go to Python Interpreter, click on plus symbol (+), type scikit-learn, select install package, refresh IDE and repeat with NLTK

---------------------------------------------------------------------------------------------------------------------------------------------------------------

You can just run the .py file from the IDE and the console will show some information on the predicted values, and in the same project folder, there will now be a csv file called "output.csv" with all the predicted values.
