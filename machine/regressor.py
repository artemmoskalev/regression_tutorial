import numpy as np
from machine.artemka_io import read_dataset, write_test_solution
from matplotlib import pyplot as pl

def main():
  X, y = read_dataset("../data/train.txt", ' ')
  T = np.loadtxt("../data/test.txt")
  print("Loading done")

  # YOUR REGRESSION CODE HERE

  # write_test_solution(your_filename_string, your_predictions_array)

if __name__ == "__main__":
  main()