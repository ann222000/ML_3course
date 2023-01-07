import pickle

# read python dict back from the file
pkl_file = open('public_tests/06_test_task6_gt/output_0.pkl', 'rb')
mydict2 = pickle.load(pkl_file)
pkl_file.close()

print(mydict2)
