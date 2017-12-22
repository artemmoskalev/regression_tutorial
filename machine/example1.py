import numpy as np
from machine.artemka_io import read_dataset, write_dataset
from matplotlib import pyplot as pl

def main():
  X, y = read_dataset("../data/regression_example1.txt", ',')
  print("Reading-in ready")

  # ONLY DATA
  # draw data with labels
  pl.scatter(X, y, s=10)
  pl.title("Data")
  pl.xlabel("City population size, million")
  pl.ylabel("Hot-dog selling van profit, K")
  pl.show()

  #####################################
  # FIND REGRESSION LINE of form y = ax
  #####################################
  w = find_weights(X, y)

  # we found line parameters, now draw line
  x_range = np.arange(5, 23, 0.5).reshape((-1, 1))
  regression_predictions = np.dot(x_range, w)

  pl.scatter(X, y, s=10)
  pl.plot(x_range, regression_predictions, color='red')
  pl.title("Linear regression $y = ax$ fit to data")
  pl.xlabel("City population size, million")
  pl.ylabel("Hot-dog selling van profit, K")
  pl.show()

  #########################################
  # FIND REGRESSION LINE of form y = ax + b
  #########################################
  Xb = np.ones((X.shape[0], 1))
  Xb = np.concatenate((X, Xb), axis=1)
  w = find_weights(Xb, y)

  # we found line parameters, now draw line
  x_range = np.arange(5, 23, 0.5).reshape((-1, 1))
  x_range_b = np.ones((x_range.shape[0], 1))
  x_range_b = np.concatenate((x_range, x_range_b), axis=1)
  regression_predictions = np.dot(x_range_b, w)

  pl.scatter(X, y, s=10)
  pl.plot(x_range, regression_predictions, color='red')
  pl.title("Linear regression $y = ax + b$ fit to data")
  pl.xlabel("City population size, million")
  pl.ylabel("Hot-dog selling van profit, K")
  pl.show()

  ########################################################
  # FIND REGRESSION POLYNOMIAL OF DEGREE 2 of form y = ax^2
  ########################################################
  X2_2 = X**2
  X2 = np.concatenate((X2_2, X), axis=1)
  w = find_weights(X2, y)

  x_range = np.arange(5, 23, 0.5).reshape((-1, 1))
  x_range_2_2 = x_range**2
  x_range_2 = np.concatenate((x_range_2_2, x_range), axis=1)
  regression_predictions = np.dot(x_range_2, w)

  pl.scatter(X, y, s=10)
  pl.plot(x_range, regression_predictions, color='red')
  pl.title("Linear regression $y = ax^2$ fit to data")
  pl.xlabel("City population size, million")
  pl.ylabel("Hot-dog selling van profit, K")
  pl.show()

  ########################################################
  # FIND REGRESSION POLYNOMIAL OF DEGREE 4 of form y = ax^4 + bx^3 + cx^2 + dx
  ########################################################
  X4_2 = X**2
  X4_3 = X**3
  X4_4 = X**4
  X4 = np.concatenate((X4_4, X4_3, X4_2, X), axis=1)
  w = find_weights(X4, y)

  x_range = np.arange(5, 23, 0.5).reshape((-1, 1))
  x_range_4_2 = x_range**2
  x_range_4_3 = x_range**3
  x_range_4_4 = x_range**4
  x_range_4 = np.concatenate((x_range_4_4, x_range_4_3, x_range_4_2, x_range), axis=1)
  regression_predictions = np.dot(x_range_4, w)

  pl.scatter(X, y, s=10)
  pl.plot(x_range, regression_predictions, color='red')
  pl.title("Linear regression $y = ax^4 + bx^3 + cx^2 + dx$ fit to data")
  pl.xlabel("City population size, million")
  pl.ylabel("Hot-dog selling van profit, K")
  pl.show()

def find_weights(X, y):
  XT = np.transpose(X)
  COV = np.dot(XT, X)
  # Xw = y; w = (X^T*X)^-1 * X^Ty
  return np.dot(np.linalg.inv(COV), np.dot(XT, y))

if __name__ == "__main__":
  main()