import numpy as np
from machine.artemka_io import read_dataset, write_dataset
from matplotlib import pyplot as pl

def main():
  X, y = read_dataset("../data/regression_example2.txt", ',')
  # 5 points into training set, 5 points into test set
  X_TRAIN = X[:int(X.shape[0]/2),]
  y_train = y[:int(X.shape[0]/2),]
  X_TEST = X[int(X.shape[0]/2):,]
  y_test = y[int(X.shape[0]/2):,]
  print("Reading-in ready")

  # ONLY TRAINING DATA
  pl.scatter(X_TRAIN, y_train, s=20, color='blue')
  pl.title("TRAINING DATA")
  pl.xlabel("x")
  pl.ylabel("y")
  pl.show()

  # TRAINING DATA and TEST DATA
  pl.scatter(X_TRAIN, y_train, s=20, color='blue')
  pl.scatter(X_TEST, y_test, s=20, color='orange')
  pl.title("TRAINING DATA AND TEST DATA")
  pl.xlabel("x")
  pl.ylabel("y")
  pl.show()

  ########################################################
  # FIND REGRESSION POLYNOMIAL OF DEGREE 4 of form y = ax^4 + bx^3 + cx^2 + dx
  ########################################################
  X_TRAIN_2 = X_TRAIN**2
  X_TRAIN_3 = X_TRAIN**3
  X_TRAIN_4 = X_TRAIN**4
  X_TRAIN_FINAL = np.concatenate((X_TRAIN_4, X_TRAIN_3, X_TRAIN_2, X_TRAIN), axis=1)
  w = find_weights(X_TRAIN_FINAL, y_train)

  xt_range = np.arange(0, 11, 0.5).reshape((-1, 1))
  xt_range_2 = xt_range**2
  xt_range_3 = xt_range**3
  xt_range_4 = xt_range**4
  xt_range_final = np.concatenate((xt_range_4, xt_range_3, xt_range_2, xt_range), axis=1)
  regression_predictions = np.dot(xt_range_final, w)

  # ONLY TRAINING DATA
  pl.scatter(X_TRAIN, y_train, s=20, color='blue')
  pl.plot(xt_range, regression_predictions, color='red')
  pl.title("TRAINING DATA WITH $y = ax^4 + bx^3 + cx^2 + dx$")
  pl.xlabel("x")
  pl.ylabel("y")
  pl.show()

  # TRAINING DATA and TEST DATA
  pl.scatter(X_TRAIN, y_train, s=20, color='blue')
  pl.scatter(X_TEST, y_test, s=20, color='orange')
  pl.plot(xt_range, regression_predictions, color='red')
  pl.title("TRAINING AND TEST DATA WITH $y = ax^4 + bx^3 + cx^2 + dx$")
  pl.xlabel("x")
  pl.ylabel("y")
  pl.show()

  ########################################################
  # FIND REGRESSION POLYNOMIAL OF DEGREE 4 of form y = ax^4 + bx^3 + cx^2 + dx WITH REGULARIZER
  ########################################################

  X_TRAIN_2 = X_TRAIN**2
  X_TRAIN_3 = X_TRAIN**3
  X_TRAIN_4 = X_TRAIN**4
  X_TRAIN_FINAL = np.concatenate((X_TRAIN_4, X_TRAIN_3, X_TRAIN_2, X_TRAIN), axis=1)
  w = find_weights_with_regularizer(X_TRAIN_FINAL, y_train, 15)

  xt_range = np.arange(0, 11, 0.5).reshape((-1, 1))
  xt_range_2 = xt_range**2
  xt_range_3 = xt_range**3
  xt_range_4 = xt_range**4
  xt_range_final = np.concatenate((xt_range_4, xt_range_3, xt_range_2, xt_range), axis=1)
  regression_predictions = np.dot(xt_range_final, w)

  # TRAINING DATA and TEST DATA
  pl.scatter(X_TRAIN, y_train, s=20, color='blue')
  pl.scatter(X_TEST, y_test, s=20, color='orange')
  pl.plot(xt_range, regression_predictions, color='red')
  pl.title("RIDGE REGRESSION $y = ax^4 + bx^3 + cx^2 + dx$ with $\lambda = 15$")
  pl.xlabel("x")
  pl.ylabel("y")
  pl.show()

  for i in range(5):
    pass


def find_weights(X, y):
  XT = np.transpose(X)
  COV = np.dot(XT, X)
  # Xw = y; w = (X^T*X)^-1 * X^Ty
  return np.dot(np.linalg.inv(COV), np.dot(XT, y))

def find_weights_with_regularizer(X, y, lmbda):
  XT = np.transpose(X)
  COV = np.dot(XT, X)
  I = np.identity(COV.shape[0])
  # Xw = y; w = (X^T*X + lmbda*I)^-1 * X^Ty
  return np.dot(np.linalg.inv(COV + lmbda*I), np.dot(XT, y))

if __name__ == "__main__":
  main()