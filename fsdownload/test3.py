
"在同个类里的一般方法中，如何调用该类的成员变量，和类方法"

"self.和 类名都可以调用类变量.类方法也是同理,cls和self是指调用这个方法的类柄"
class Person:

    name = "gaoyan"

    @classmethod
    def hello(cls):
        print(cls.name)

    def f(self):
        # print(self.name)
        # print(Person.name)
        self.hello()
        Person.hello()

person = Person()
person.f()
