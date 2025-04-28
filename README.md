# Custom Linear regression
In this repository, we compare a custom linear regression model built from scratch with, the standard Sklearn's LinearRegression.

Two methods are implemented but with a single adoption of the one with the best MSE according to our data set and how we managed to get the better predictions. These are:

* Gradient descent method (experimental)
* Analytical - Closed form (adopted)

Both methods are explained in detail [here](https://www.cs.toronto.edu/~rgrosse/courses/csc311_f20/readings/notes_on_linear_regression.pdf)


## Setup And Run
To run the comparison, run the following commands
* Create a new conda environment using the requirement file dependencies file 

```bash
conda create --name myenv --file requirements.txt
```
* activate the conda environment 
```bash
conda activate myenv
```
* From the root directory, run the following
```bash
python run.py
```

## Evluation
The mse for both models are show below using gradient descent.

| Model type  | Train               | Test               |
|---|------------------------|------------------------|
| Custom| 2.8555451254704792e+79 | 2.7462612127205875e+79 |
| Sklearn | 0.23575564509792812    | 0.19591829337932573    |

Note: These values are prior to changing slightly during next run.

# Discussion
From obersvation of the `mse`(Mean Square Error) of the test set, we notice that the custom model is far from making accurate predictions. This could be improved by:

* Exploring other methods such as closed-form which provides a clear formula for calculaitng the optimal B which minimizes the
  on the training set X.

Comparing the closed-form method with the LinearRegression from Scikit learn, we have

| Model type  | Train               | Test               |
|---|------------------------|------------------------|
| Custom| 0.6072736646666331  | 0.5924563815227322 |
| Sklearn | 0.526420936861469 | 0.5162680273701334|

## Conclussion

Based on the results above, we can be certain that Sklearn implements a function which approximates almost
the same values as the close-form for its predictions.
