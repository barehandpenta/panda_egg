import random as rd
from common.common import open_csv_file, copy_paste, my_shuffle

# Function to distribute 1200 data from total examples to train examples and test examples:
def data_distribute(total_csv, train_csv, test_csv):
    # Load examples data:
    total_data = open_csv_file(total_csv, 'r')# TOTAL EXAMPLE
    train_data = open_csv_file(train_csv, 'w')# TRAIN EXAMPLE
    test_data = open_csv_file(test_csv, 'w')# TEST EXAMPLE
    # Create list contains indexes of data:
    total_list = []
    shred_list = []
    # Total list contains index from 0 - 1199
    for i in range(1200):
        total_list.append(i)
    # Shred list contains index of 200 random numbers from 0 - 1199
    shred_list = rd.sample(range(1200), 200)
    # Randomly take 200 index out of total index list:
    ii = 0
    for i in shred_list:
        total_list.pop(i - ii)
        ii += 1
    # Shuffle 2 list above to get a more random train and test list:
    test_list = my_shuffle(shred_list)
    train_list = my_shuffle(total_list)
    print(test_list)
    # Write data to CSV file using 2 index list and TOTAL EXAMPLE DATA:
    for i in test_list:
        test_data.writerow(total_data[i])
    for i in train_list:
        train_data.writerow(total_data[i])

def main():
    # Sequently distribute stand, sit and lie data set into train and test:
    data_distribute('stand.csv', 'stand_train.csv', 'stand_test.csv')
    data_distribute('sit.csv', 'sit_train.csv', 'sit_test.csv')
    data_distribute('lie.csv', 'lie_train.csv', 'lie_test.csv')
    # The last steps of data preparing:
    # The goals is to combined 3 train data files to konel_egg_train.csv and the same for the konel_egg_test.csv
    # Before writing into the files, we may want to shuffle the data for better performing when we train the network

    # Load 6 CSV file which are just distributed above as read-only type:
    stand_train = open_csv_file('stand_train.csv', 'r')
    stand_test = open_csv_file('stand_test.csv', 'r')
    sit_train = open_csv_file('sit_train.csv', 'r')
    sit_test = open_csv_file('sit_test.csv', 'r')
    lie_train = open_csv_file('lie_train.csv', 'r')
    lie_test = open_csv_file('lie_test.csv', 'r')
    # Load 2 last CSV file for the network as write type:
    konel_egg_train = open_csv_file('konel_egg_train.csv', 'w')
    konel_egg_test = open_csv_file('konel_egg_test.csv', 'w')
    train_data = []
    test_data = []
    # Copy train data of STAND SIT LIE and paste to final train data:
    copy_paste(stand_train, train_data)
    copy_paste(sit_train, train_data)
    copy_paste(lie_train, train_data)
    # Do the same for test data:
    copy_paste(stand_test, test_data)
    copy_paste(sit_test, test_data)
    copy_paste(lie_test, test_data)
    # Shuffle up data for better performance in the Network:
    train_data = my_shuffle(train_data)
    test_data = my_shuffle(test_data)
    # Last step, writing data into CSV files:
    for i in range(len(train_data)):
        konel_egg_train.writerow(train_data[i])
    for i in range(len(test_data)):
        konel_egg_test.writerow(test_data[i])



if __name__ == '__main__':
    main()




