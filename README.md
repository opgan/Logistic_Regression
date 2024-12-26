[![Python Install/Lint with Github Actions](https://github.com/opgan/Logistic_Regression/actions/workflows/main.yml/badge.svg)](https://github.com/opgan/Logistic_Regression/actions/workflows/main.yml)

# Logistic_Regression
Python implementation of binary classification with linear regression

## Set up working environment
* virtual environment: ```virtualenv ENV```
    - remove directory: ``` rm -r hENV```
    - add ```source ENV/bin/activate``` in .bashrc file
    - edit bashrc file ```vim ~/.bashrc``` and goes to the last line: ```shift + g``` 
* Makefile for make utility : ``` touch Makefile ```
    - format codes ``` make format ```
    - run lint ``` make lint ```
    - run main.py ``` make run```
* Requirements.txt for package installation : ``` touch Requirement.txt ```
    - find out package version: ```pip freeze | less```
    - install packages: ``` make install ```
    - ``` pip install <name> ```
* Project folders:
   - create directory ``` mkdir myLib ```
   - rename a file: ```mv oldfilename newfilename```
   - myLib folder contains images of plots
   - tests folder contains test scripts to assess the correctness of some functions
   - plots folder planar_utils provide various useful functions used in this assignment
* Logging
    - info.log file contains information of prediction results
* Running Main.py from ipython or Command Line Interface (CLI)
  - ipython: ```run main.py``` or CLI:  ```python main.py injest 5 ```  ```python main.py modeling ``` ``` python main.py predict-test 5 ``` ``` python main.py predict-unseen tree.jpg```


## Machine Learning Procedure
* Injest the dataset ("data.h5") containing: 
    - a training set of m_train images labeled as cat (y=1) or non-cat (y=0) 
    - a test set of m_test images labeled as cat or non-cat 
    - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB)
        ```
        Number of training examples: m_train = 209
        Number of testing examples: m_test = 50
        Height/Width of each image: num_px = 64
        Each image is of size: (64, 64)
        Original train_set_x shape: (209, 64, 64, 3)
        Original train_set_y shape: (209,)
        Original test_set_x shape: (50, 64, 64, 3)
        Original test_set_y shape: (50,)
        ```
    - Preprocess data: flattern x and reshape y
        ```
        Flatten and standardized train_set_x shape: (12288, 209)
        Flatten and standardized  test_set_x shape: (12288, 50)
        Row-wise train_set_y shape: (1, 209)
        Row-wise  test_set_y shape: (1, 50)
        ```
* Implement linear regression
    - Use the non-linear sigmoid activation function  
    - Compute the loss
    - Implement forward and backward propagation
    - Initialize the parameters of the model 
    - Learn the parameters for the model by minimizing the cost
    - Use the learned parameters to make predictions (on the test set) 
    - Analyse the results and conclude
    - Calculate current loss (forward propagation)
    - Calculate current gradient (backward propagation)
    - Update parameters (gradient descent)
