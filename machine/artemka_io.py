import numpy as np

def read_dataset(filename, custom_delimiter):
  XA = np.loadtxt(filename, delimiter=custom_delimiter)
  # all columns except last are features
  X = XA[:, :-1]
  # last column is labels
  y = XA[:, -1]
  return X, y

def write_dataset(filename, labels):
  np.savetxt("../output/" + filename, labels)

def write_test_solution(filename, labels):
  f = open("../output/" + filename, "w")
  f.write("Sample_id,Sample_label\n")
  for i in range(labels.shape[0]):
    f.write(str(i+1) + "," + str(int(labels[i])) + "\n")
  f.close()