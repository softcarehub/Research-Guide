# Table of Contents 
- [Heat Diseases working step](#heat-diseases-working-step)
- [Ensemble methods in machine learning](#ensemble-methods-in-machine-learning)

## Basic understanding
- what's abstract: a concise summary of a research paper or entire thesis. 
- what's Introduction: 
- What is proposed methodology in thesis?
  - Methodology refers to the overall rationale and strategy of your thesis project. It involves studying the theories or principles behind the methods used in your field so that you can explain why you chose a particular method for your research approach.

## Heat Diseases working step
1. Machine Learning Model
- going to build a logistic regression  machine learning model here. 
- so build the model need patient data set 
- once dataset is ready then have to do preprocessing of the dataset 
- then spitting the data set into two parts one for training and one for testing dataset.
- the we have create logistic regression model from the sklearn library 
- then training the model using the training data set. 
- once the model is trained then we would be testing our model with the test data
- so once the testing is completed then our model is ready to be predict the values
- so once our model is ready to predict the values we will be deploying it as a web application 

2. Tools 
- Numpy 
- Pandas 
- Sci-kit Learn
- Imblearn

## Ensemble methods in machine learning
### What is Ensemble method?
- ![What's SVM?](images/ensemble.png)
- `combine multiple models` (holding with best accuracy not like it) to improve the accuracy of results. 
- In ensemble methods in machine learning, the idea is to combine multiple models to improve overall predictive performance. Ensemble methods can be broadly categorized into two types: bagging and boosting. Ensemble learning helps improve machine learning results by combining several models. This approach allows the production of better predictive performance compared to a single model. Basic idea is to learn a set of classifiers (experts) and to allow them to vote. Advantage : Improvement in predictive accuracy.
### Example of some ensemble methods:
  - `Bagging:` Reduces variance and prevents over fitting
     - In bagging, multiple instances of the same base model are trained independently on different subsets of the training data (usually sampled with replacement).
     - Examples include Random Forest, Bagged Decision Trees.
  - `Stacking:` Builds a new model based on the combined output of multiple models
     - ![What's SVM?](images/stacking.png)
     - Idea: Stacking, also known as Stacked Generalization, involves training a meta-model that combines the predictions of multiple base models. The base models are trained independently, and their outputs serve as input features for the meta-model.
     - Process:Train multiple diverse base models on the training data. Use the predictions of these base models as input features for a higher-level model (meta-model). The meta-model learns to combine the predictions of base models to make the final prediction.
  - `Boosting:` Reduces bias
     - Boosting algorithms train a series of weak learners sequentially, with each learner trying to correct the errors of its predecessor.
     - Examples include AdaBoost, Gradient Boosting, XGBoost, LightGBM.
  - `Gradient Boosting:` Improves each weak learner
     - Idea: Gradient Boosting is an ensemble technique where weak learners (usually decision trees) are trained sequentially. Each new tree corrects the errors of the combined ensemble.
     - Process: Train a weak learner (tree) on the data. Compute the errors made by the ensemble so far. Build a new weak learner to correct these errors. Repeat until a predefined number of weak learners is reached. Examples: XGBoost, LightGBM, AdaBoost with decision trees.
  - `AdaBoost:Adaptive Boosting` Reassigns weights to each instance, with higher weights for misclassified instances
     - Idea: AdaBoost is a boosting algorithm that assigns weights to the observations. It focuses on the mistakes made by the previous models and gives higher weights to misclassified data points.
     - Process:Train a weak learner.Increase the weights of misclassified observations.Train another weak learner with updated weights.Repeat until a predefined number of weak learners is reached. Combine the weak learners into a strong model.
     - Adaboost falls under the supervised learning branch of machine learning. This means that the training data must have a target variable. Using the adaboost learning technique, we can solve both classification and regression problems.
     - Example: AdaBoost with decision trees.
  - `Voting:` Trains multiple model algorithms and makes them "vote" during the prediction phase
     - Idea: Voting, or Majority Voting, combines predictions from multiple models and selects the class that receives the majority of votes.
     - Process: Train multiple diverse base models independently. When making predictions, each model votes for a class.The class with the majority of votes is the final prediction.
     - Types: Hard Voting: Each model contributes one vote. Soft Voting: Models provide probabilities, and the class with the highest average probability is selected. Example: sklearn's VotingClassifier.
-  Other ensemble methods include: Bootstrapping, Heterogeneous ensemble. When choosing between these methods, consider the characteristics of your data and the problem at hand. Stacking requires more computation but often leads to improved performance. Gradient Boosting and AdaBoost are effective for handling weak learners, and Voting is a simple yet powerful ensemble method. Experimenting with different approaches is key to finding the best ensemble strategy for a particular problem.
### why AdaBoost is considered an ensemble method?
- `Multiple Weak Learners:` AdaBoost doesn't use just one decision tree; it uses multiple weak learners (decision trees). Each weak learner is trained sequentially, and its performance is weighted based on the errors made by the previous learners. Weak learners are used in ensemble methods, where multiple weak learners are combined to create a strong learner that can make more accurate predictions.
### When and How choosing models for ensemble method?
- `Introduction:` it's common to use diverse models that have different strengths and weaknesses. The idea is that errors made by one model might be compensated by another model. Here are some guidelines:
- `Diversity:` The models you choose should be diverse, i.e., they should make different types of errors. For example, combining a decision tree with a linear model could be more effective than combining two similar decision trees.
- `Model Types:` You can mix models of different types or architectures. For example, combining a decision tree with a neural network.
- `Hyperparameter Tuning:` Train each base model with different hyperparameters to maximize their individual performance.
- `Avoid Perfect Correlation:` If two models are perfectly correlated (i.e., they make the exact same predictions), ensembling them won't provide much benefit. Diversity is key.
- `Performance Metrics:` Choose models that perform well on the specific metric you care about. If one model is significantly better than others based on the metric of interest, it might make sense to give it more weight in the ensemble.
- `Consider Computation Time:` Ensemble methods can be computationally expensive, especially if you're training a large number of models. Consider the trade-off between computational cost and potential performance improvement.

### Motivation
- Ultimately, the effectiveness of ensembling depends on the dataset, the choice of models, and the specific problem you are trying to solve. It's often a good idea to experiment with different combinations and observe how the ensemble performs on validation or test data.
 

## algorithm
1. SVM
 ### Step 1: Introduction to SVM
- What's SVM?
![What's SVM?](images/whatsSVM.png)
- `Objective:` SVM is a machine learning algorithm used for classification and regression.
![SVM Objective](images/SVM.png)
- `Hyperplane:` A line that separates data into classes. a hyperplane is a decision boundary that separates two classes. 
- `idea`It finds the optimal hyperplane that best separates different classes in the feature space.
- `Support Vectors:` Data points that determine the position and orientation of the hyperplane.
-` Margin:` The distance between the hyperplane and the nearest data point of either class.
### step 2: Types of SVM
- Types of SVM: 1. liner and 2. Non-linear
 - Linear SVMs are used for linearly separable data, whereas nonlinear SVMs and kernel SVMs are used for non-linearly separable data.
 ![Linear and Non-Linear](images/linear.png.png)
### Step 3: Linear SVM
- `Linear Separation:` In a 2D space, it's like finding the best straight line to separate two classes.
- `Equation of Hyperplane:` w*x+b=0. where w is the weight vector, x is the input vector, and b is the bias.
### Step 4: Non-Linear SVM
- `Kernel Trick:` When data is not linearly separable, introduce the concept of transforming data using a kernel function.
- `Common Kernels:` Polynomial, Radial Basis Function (RBF/Gaussian), Sigmoid, etc.
### Step 5: Soft Margin SVM
- `Reality of Data:` Many datasets are not perfectly separable.
- `C Parameter: `It controls the trade-off between having a smooth decision boundary and classifying the training points correctly.
### Step 6: Code Implementation
- `Import Library:` from sklearn import svm
- Create SVM Model: `model = svm.SVC(kernel='linear', C=1)`
- Fit the Model:` model.fit(X_train, y_train)`
- Make Predictions: `predictions = model.predict(X_test)`
### Step 7: Evaluation
- `Accuracy: `Percentage of correctly classified instances.
- `Confusion Matrix:` Breakdown of true positive, true negative, false positive, and false negative.
### Step 8: Real-World Applications
- `Image Recognition:` SVM can be used for facial recognition.
- `Text Classification:` Classifying documents into categories.
- `Medical Diagnosis: `Identifying disease based on patient data.
### Step 9: Advantages and Disadvantages
- `Advantages: `Effective in high-dimensional spaces, memory-efficient.
- `Disadvantages:` Not suitable for larger datasets, sensitive to noise.
### Step 10: Resources for Further Learning