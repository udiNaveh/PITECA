from model.models import *
from sharedutils.subject import *
from sharedutils.constants import *
import tensorflow as tf
from abc import ABC, abstractmethod


class Person(ABC):
    def __init__(self, name, phone):
        self.name = name
        self.phone = phone


class Teenager(Person):
    def __init__(self, name, phone, website):
        super(Teenager, self).__init__(name, phone)
        self.website = website


def test_inheritence():
    teen = Teenager("guy", "123", "walla")
    print(teen.website, teen.phone, teen.name)


def run_simple_model():
    subj = Subject(subject_id= '001',
                   input_path= '',
                   output_path= '',
                   features_path= '')
    tasks = [Task.MATH_STORY, Task.PUNISH_REWARD]
    my_model = LinearModel(tasks)
    print(my_model.tasks)
    my_model.predict(subj)


class C:
    __classprop = 4

    def __init__(self, value):
        self.value = value

    def do_something(self):
        C.classprop



if __name__ == "__main__":
    print ("udi's experiments main called")
    run_simple_model()



