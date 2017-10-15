from database import Database
import helper


for i in range(1, 4):
    db = Database(0)
    db.compute_test_set()
    test_set = db.get_test_set()
    train_list = db.get_train_list()
    helper.write("test_set"+str(i), test_set)
    helper.write("train_set"+str(i), train_list)