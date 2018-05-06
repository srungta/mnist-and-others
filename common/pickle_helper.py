import pickle

def read_from_pickle(pickle_file):
    pickle_file = open(pickle_file, "rb")
    emp = pickle.load(pickle_file)
    pickle_file.close()
    return emp