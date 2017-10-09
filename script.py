from database import Database
import helper

for i in xrange(3):
    db = Database(True)
    test_set = db.get_test_set()
    train_list = db.get_train_list()
    helper.write("test_set"+str(i), test_set)
    helper.write("train_set"+str(i), train_list)