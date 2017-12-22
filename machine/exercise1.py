import numpy as np
from machine.artemka_io import read_dataset, write_dataset
from matplotlib import pyplot as pl

def main():
  X, y = read_dataset("../data/regression_example1.txt", ',')
  print("Reading-in ready")

  #####################################
  # FIND REGRESSION LINE of form y = ax^3 + bx^2 + cx + d
  #####################################

  x_range = []
  regression_predictions = []

  # YOUR CODE STARTS HERE

  # YOUR CODE ENDS HERE

  pl.scatter(X, y, s=10)
  pl.plot(x_range, regression_predictions, color='red')
  pl.title("Linear regression $y = ax^2 + bx^2 + cx + d$ fit to data")
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