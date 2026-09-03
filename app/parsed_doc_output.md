P.Ramkumar_ BAI602

BIVIARIATE DATA AND MULTIVARIATE DATA 

Bivariate data involves two variables.

goal of bivariate analysis is to explore the relationship between variables.

This relationship can help in 

comparisons,

identifying causes,

further exploration of the data.

P.Ramkumar_ BAI602 

Table 2.3, with data of the temperature in a shop and sales of sweaters. • Bivariate Data involves two variables. • Bivariate data deals with causes of relationships.  • The aim is to find relationships among data.

P.Ramkumar_ BAI602 

P.Ramkumar_ BAI602

A scatter plot is a useful  graphical method for visualizing  bivariate data.

P.Ramkumar_ BAI602 

The key features of a scatter plot are: 

Strength: Indicates how closely the data points fit a pattern  or trend.

Shape: Helps in identifying the type of relationship (linear,  quadratic, etc.).

Direction: Shows whether the relationship is positive,  negative, or neutral.

Outliers: Helps identify any points that deviate significantly   from the trend.

P.Ramkumar_ BAI602 

There are various statistical measures to describe the  relationship between two variables. 

Two important bivariate statistics are  

1. Covariance 

2. Correlation. 

P.Ramkumar_ BAI602 

Covariance 

Covariance measures the joint variability of two random  variables.

It tells you whether an increase in one variable results  in an increase or decrease in the other variable.

Mathematically, the covariance between two variables X  and Y is defined as:

P.Ramkumar_ BAI602 

The formula for the covariance between two datasets  X and Y is: 

where: 

X and y are individual data points from sets X and Y. • E(X)and E(Y) are the means of X and Y, respectively. • N is the number of data points.

P.Ramkumar_ BAI602 

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

Covariance values: 

 Positive covariance: As one variable increases, the other  variable also increases. 

Negative covariance: As one variable increases,the other  variable decreases. 

Zero covariance: No linear relationship between the variables.

P.Ramkumar_ BAI602 

Correlation 

covariance measures the direction of the relationship.

correlation quantifies the strength of the relationship  between two variables.

P.Ramkumar_ BAI602 

The most common measure of correlation is the Pearson  correlation coefficient.

P.Ramkumar_ BAI602 

Find the correlation coefficient of data  X={1,2,3,4,5} AND Y={1,4,9,16,25} The mean value of 

 X =15/5=3 

 Y =55/5=11 

Standard deviation of 

X =1.41 

Y = 8.6486

P.Ramkumar_ BAI602 

Multivariate Statistics 

Multivariate data refers to data that involves more than two  variables. 

Machine learning, most datasets are multivariate. • The goal of multivariate analysis is to understand  relationships among multiple variables simultaneously.

P.Ramkumar_ BAI602 

mean and variance for each attribute column are: Attribute 1: Mean = 2.00, Variance = 0.67 Attribute 2: Mean = 5.00, Variance = 0.67 Attribute 3: Mean = 1.33, Variance = 0.22 

P.Ramkumar_ BAI602 

The mean vector is used to represent the mean of multiple  variables. 

the covariance matrix represents the variance and  relationships among all variables. 

The mean vector is also known as the centroid. 

The covariance matrix is also referred to as the dispersion  matrix.

P.Ramkumar_ BAI602 

Multivariate analysis techniques include: 

1. Regression Analysis 

2. Principal Component Analysis (PCA) 3. Path Analysis

P.Ramkumar_ BAI602 

Heatmap 

graphical representation of a 2D matrix  • Takes matrix as input and colors it.

values are represented by colors.

The darker color indicate larger value

The lighter color indicate smaller value.

Advantage : 

Human perceive color well

By color shaping larger values can be perceived well.

P.Ramkumar_ BAI602 

Example:  

Action heat map in football

P.Ramkumar_ BAI602 

Patient data highlighting weight count

P.Ramkumar_ BAI602 

Applications: 

 Heatmaps are useful for visualizing complex data like  • traffic patterns 

patient health data,

 where you can easily identify regions of higher or lower values. 

Example: 

In vehicle traffic data, regions with heavy traffic are highlighted  with dark colors, making it easy to spot problem areas.

P.Ramkumar_ BAI602 

Pairplot (or Scatter Matrix) 

A pairplot (or scatter matrix) is a matrix of scatter plots that  shows relationships between every pair of variables in a  multivariate dataset.

This method allows you to visually examine correlations or  relationships between variables.

A random matrix of three columns is chosen and the  relationships of the columns is plotted as a pairplot (or  scattermatrix)

P.Ramkumar_ BAI602 

Visual Layout: Each  

scatter plot in the  

matrix shows the  

relationship  

between two  

variables. 

Usefulness: By examining the pairplot, you can easily identify  patterns, correlations,or clusters among the variables.

P.Ramkumar_ BAI602 

Essential Mathematics for Multivariate Data 

These include concepts from  

1. Linear Algebra, 

2. Statistics, 

3. Probability, 

4. Optimization.

P.Ramkumar_ BAI602 

Linear Algebra 

Linear algebra is crucial in machine learning as it provides the  tools for dealing with data in the form of vectors and matrices.  Here's a breakdown of important topics: 

Vectors: A vector is an ordered list of numbers. It can  represent data points or features of an observation in a  multivariate dataset.

P.Ramkumar_ BAI602 

Dot product and cross product are used to compute  projections and angles between vectors. 

Matrices: A matrix is a 2D array of numbers. In machine  learning, matrices often represent data where rows are  instances and columns are features.

P.Ramkumar_ BAI602 

Determinants and Inverses: 

The determinant of a matrix tells us if the matrix is  invertible (non-singular).

The inverse of a matrix is used to solve linear systems of  equations.

Singular Value Decomposition (SVD): 

 This is a factorization method used in PCA and other  dimensionality reduction techniques to decompose a matrix  into singular values and vectors.

P.Ramkumar_ BAI602 

Statistics 

Statistics is key to understanding the relationships  between different variables in multivariate data. Key  concepts include: 

Mean and Variance: Measures of central tendency  (mean) and spread (variance) are essential to  understanding the distribution of each variable. 

Covariance: Covariance measures the relationship  between two variables. A positive covariance indicates  that as one variable increases, the other tends to increase.

P.Ramkumar_ BAI602 

Correlation: Correlation is a normalized measure of  covariance that indicates the strength and direction of the  relationship between two variables. 

Multivariate Normal Distribution: Many machine learning  algorithms assume that the data follows a multivariate normal  distribution, which extends the idea of normal distribution to  more than one variable. 

Principal Component Analysis (PCA): PCA is used to reduce  the dimensionality of the dataset while retaining as much  variance as possible. It uses eigenvectors and eigenvalues to  identify the principal components.

P.Ramkumar_ BAI602 

Probability 

Probability theory underpins the concept of uncertainty, which  is inherent in real-world data: 

Random Variables: 

 A random variable represents a quantity whose value is subject  to chance. In multivariate data, we deal with vectors of random  variables. 

Probability Distributions:  

These describe the likelihood of various outcomes. Common  distributions in machine learning include the normal distribution  and the multinomial distribution.

P.Ramkumar_ BAI602 

Optimization 

Optimization is key to finding the best model for multivariate  data. Many machine learning algorithms are formulated as  optimization problems. 

Gradient Descent: An iterative optimization algorithm used to  minimize a cost function (such as in linear regression or neural  networks).

P.Ramkumar_ BAI602 

Convex Optimization:  

Involves minimizing convex functions, and plays a significant  role in machine learning, as many cost functions are convex. Lagrange Multipliers:  

Used for optimizing functions subject to constraints, which is  often seen in constrained optimization problems in machine  learning.

P.Ramkumar_ BAI602 

Multivariate Analysis 

Multivariate Regression: This is the extension of linear  regression to predict multiple dependent variables using a set  of independent variables. 

Multivariate Analysis of Variance (MANOVA): An extension of ANOVA used when there are two or more  dependent variables. It tests for differences between groups. 

Factor Analysis: 

A method for identifying the underlying relationships between  observed variables. It’s often used in exploratory data  analysis.

P.Ramkumar_ BAI602 

Linear Systems and Gaussian Elimination for Multivariate  Data 

A linear system of equations is a group of equations with  unknown variables. 

Let Ax=y then the solution x is given as: 

x = y/A = A-1y 

If there is a unique solution, then the system is called consistent  independent.  

If there are various solutions, then the system is called  consistent dependent. 

If there are no solutions and the equations are contradictory,  then the system is called inconsistent.

P.Ramkumar_ BAI602 

For solving large number of system of equations, Gaussian  elimination can be used.  

The procedure for applying Gaussian elimination is given as  follows: 

1. Write the given matrix. 

2. Append vector y to the matrix A. This matrix is called  augmentation matrix. 

3. Keep the element a11 as pivot and eliminate all a11 in  second row using the matrix operation,

P.Ramkumar_ BAI602 

P.Ramkumar_ BAI602

To facilitate the application of Gaussian elimination method  the following row operations are applied 

applied: 

1.Swapping the rows 

2.Multiplying or dividing row by a constant 

2.Replacing the row by adding or subtracting a multiplayer of  another row to it.

P.Ramkumar_ BAI602 

Echelon Form 

https://youtu.be/zksRGHYD76g?si=TTJbnmYQMUPfCUex

Echelon form means that the matrix is in one of two states: 

1. Row echelon form. 

2. Reduced row echelon form. 

This means that the matrix meets the following three  requirements: 

1. The first number in the row (called a leading coefficient) is 1.  2. Every leading 1 is to the right of the one above it. 3. Any non-zero rows are always above rows with all zeros. 

P.Ramkumar_ BAI602 

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

Are the following in echelon form

P.Ramkumar_ BAI602 

Are the following in echelon form

P.Ramkumar_ BAI602 

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

Reduced row echelon form 

Only if it is in row echelon form  

All its pivots are equal to 1

The pivots are the only non-zero entries of the basic columns.

 It has one zero row (the third)which is below the non-zero rows.   The first and the second row are non-zero, but have a pivot   (A11 and A23 respectively).  

 The two pivots are equal to 1 and they are the only non-zero   entries in their respective columns.

P.Ramkumar_ BAI602 

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

Step 4: Extract the solutions

P.Ramkumar_ BAI602 

LU decomposition 

Also known as LU factorization.

It’s a method in linear algebra.

Decomposes a square matrix into the product of a lower  triangular matrix (L) and an upper triangular matrix (U).

P.Ramkumar_ BAI602 

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

P.Ramkumar_ BAI602

Machine learning &importance of Probability and statics 

Probability theory underpins the concept of uncertainty,  which is inherent in real-world data: 

Random Variables: Represents a quantity whose value is  subject to chance. 

In multivariate data, we deal with vectors of random variables.

P.Ramkumar_ BAI602 

Probability Distributions: 

 These describe the likelihood of various outcomes. Common distributions in machine learning include the 1.Normal distribution 

2.The multinomial distribution.

P.Ramkumar_ BAI602 

Discrete probability distribution 

A probability distribution that gives the finite trials of a discrete  random variable at a given point in time is called a discrete  probability distribution. 

Conditions for the discrete probability distribution are: Probability of a discrete random variable lies between 0 and 1: 0  ≤ P (X = x) ≤ 1 

Sum of Probabilities is always equal to 1: ∑ P (X =x) = 1

P.Ramkumar_ BAI602 

Let two coins be tossed then the probability of getting a tail is  an example of a discrete probability distribution.  The sample space for the given event is {HH, HT, TH, TT} and  X be the number of tails then, the discrete probability  distribution table is given by: 

x 0 {HH} 1 {HT, TH} 2 {TT}

P.Ramkumar_ BAI602 

Continuous probability distributions (CPDs)  Are probability distributions that apply to continuous random  variables.  

It describes events that can take on any value within a specific  range,  

Example: 

The height of a person

The amount of time it takes to complete a task.

P.Ramkumar_ BAI602 

Example :A model to predict the price of a car.  You have data on various factors like mileage, year, and  brand. But how do you account for the fact that prices can  vary continuously? 

 This is where continuous distributions come to the rescue!  By fitting a suitable distribution to the price data, you can  estimate the probability of a car with specific features  falling within a certain price range.

P.Ramkumar_ BAI602 

Density Estimation  

Probability Density Function (PDF) tells us how likely  different outcomes are for a continuous variable. 

Maximum Likelihood Estimation helps us find the  best-fitting model for the data we observe. 

P.Ramkumar_ BAI602 

Density Estimation? 

Density estimation aims to find the probability distribution (or  density) of a dataset, allowing us to understand how likely  different values are.  

It's an unsupervised learning technique(doesn't require labeled  data).  

Common techniques include: 

Histograms: A basic method for visualizing data distribution  by grouping data into bins.  

Kernel Density Estimation (KDE): A more flexible method  that uses a "kernel" function to estimate the density, smoothing  out the data.  

Mixture Models: Assumes the data is generated from a  P.Ramkumar_ BAI602 

combination of simler distributions. 

Parametric Density Estimation is a statistical approach used  to estimate the probability density function (PDF) of a dataset  by assuming that the data follows a known distribution with a  finite set of parameters. 

Non-Parametric Density Estimation is a method used to  estimate the probability density function (PDF) of a dataset  without assuming an underlying parametric distribution. 

Instead of defining a fixed set of parameters, these methods let  the data determine the shape of the distribution.

P.Ramkumar_ BAI602 

FEATURE ENGINEERÏNG AND DIMENSIONALITY   REDUCTION TECHNIQUES 

Features are attributes.  

Feature engineering is about determining the subset of  features that form important part of the input that improves the  performance of the model, be it classification or any other model  in machine learning.

P.Ramkumar_ BAI602 

Feature engineering deals with two problems  

1. Feature Transformation 

2. Feature Selection. 

Feature transformation :  

Extraction of features and creating new features that may be  helpful in increasing performance.  

Eg:The height and weight may give a new attribute called Body  Mass Índex (BMI).

P.Ramkumar_ BAI602 

Feature subset  

It reduces the dataset size by removing irrelevant features  and constructs minimum set of attributes for machine  learning.

Focuses on selection of features to reduce the time but not at  the cost of Reliability.

Components that do not contribute much can be deleted. • If the dataset has n attributes, then time complexity is  extremely high as n dimensions need to be processed for the  given dataset.

For n attributes there are there are 2n possible subsets

P.Ramkumar_ BAI602 

The features can be removed based on two aspects: 1.Feature relevancy  

Some features contribute more for classification than other  features.

For example, a mole on the face can help in face detection  than common features like nose.

In simple words, the features should be relevant

P.Ramkumar_ BAI602 

2.Feature redundancy – 

Some features are redundant.

Eg: when a database table has a field called Date of  birth, then age field is not relevant as age can be  computed easily from date of birth.  

It helps in removing the column age that leads to  reduction of dimension one.

P.Ramkumar_ BAI602 

Wrapper-based methods use classifiers to identify the best  features.

These are selected and evaluated by the learning  algorithms.

Some of the important algorithms that fall under this category. 1. Stepwise Forward Selection 

2. Stepwise Backward Elimination 

3. Principal Component Analysis

P.Ramkumar_ BAI602 

Stepwise Forward Selection 

Is a feature selection technique

Start with an empty model

Iteratively add the most significant variables until no further  improvement in model performance is observed.

P.Ramkumar_ BAI602 

Example : 

A company wants to predict whether an employee will leave the  organization based on various factors such as  

Age

Salary

Work Experience

Job Satisfaction

Commute Time

Number of Promotions

Training Hours

P.Ramkumar_ BAI602 