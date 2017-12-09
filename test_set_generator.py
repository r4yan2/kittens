from database import Database
import helper


for i in range(4, 7):
    db = Database(0)
    db.compute_test_set_v1()
    test_set = db.get_test_set()
    train_list = db.get_train_list()
    helper.write("test_set"+str(i), test_set, '\t')
    helper.write("train_set"+str(i), train_list, '\t')
