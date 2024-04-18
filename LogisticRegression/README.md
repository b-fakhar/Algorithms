# K-fold cross validation technique in Python

# Table of Contents
1. [Objective](#objective)
2. [Instructions](#instructions)
4. [Description](#description)

-------------------------------------------------------------------------------------------
## 1. Objective
- Write K-fold cross validation technique in Python.


## 2. instructions

##### 2.1 Conditions
- Every element from the sample should be used exactly once in testing sets.
- Every element from the sample should be used at least once in training sets.
- The sizes of the pairs of testing sets may differ by no more than 1.

##### 2.2 Inputs
- An array of integers indices, which contains the indices of elements in data sample.
- An integer K which, refers to the number of folds.

##### 2.3 Output
- An array that contains the arrays of folds.

##### 2.4 Assumptions
- N is an integer within the range [2, ..., 100].
- K is an integer within the range [2, ..., N].
- Each element of array indices is an integer within the range [0, ..., 100,000].
- The elements of indices are all distinct.

##### 2.5 Example
- Given indices = [1,2,3] and K = 2;
    - One of the accepted results may be [[1,2],[3],[3],[1,2]]
i.e. first fold contains [1,2] as the training set and [3] as the testing set and second fold contains [3] as the training set and [1,2] as the testing set.

## 3. Description

###### Let's assume:

_**jump**_: Result of the division between N and K after rounding up.

_**rem**_: Remainder of the division between N and K.

###### Then, some possible test cases can be summarized as follows:

![image](https://user-images.githubusercontent.com/59096353/134703667-fb03cfbc-fbe2-4a32-a435-dc438774d4fa.png)




