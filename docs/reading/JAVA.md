
## 数据库设计三范式

- 第一范式（1NF）：原子性（存储的数据应该具有“不可再分性”）
- 第二范式（2NF）：唯一性 (消除非主键部分依赖联合主键中的部分字段)（一定要在第一范式已经满足的情况下）
- 第三范式（3NF）：独立性，消除传递依赖(非主键值不依赖于另一个非主键值)

## cache

- 缓存失效
- 缓存穿透
- 缓存雪崩
- 缓存热点
- 数据不一致
- 数据并发竞争
- Hot Key
- Big Key

## GaussDB

2024-07-04 工作小结:

- 不同会话不能同时查询、修改等，会阻塞；
- 执行事务有一条失败，就会整个回滚；可使用savepoint设置达到Oracle类似效果
- 默认空间为目录，与Oracle不一样
- 空串“”和null是否一致（不确定），Oracle是一样的，MySQL是不一样的
- 建议严格astore，使用vaccum对无用数据进行清除
- DDL自身不会提交，未提交的DDL操作会阻塞表或分区的操作；执行DDL时不会提交当前连接中的事务

## 重载与重写

### 重载 Overload

- 在一个类中，同名的方法如果有不同的参数列表（参数类型不同、参数个数不同甚至是参数顺序不同）则视为重载。
- 同时，重载对返回类型没有要求，可以相同也可以不同，但不能通过返回类型是否相同来判断重载。

```java
public class Father {

    public static void main(String[] args) {
        // TODO Auto-generated method stub
        Father s = new Father();
        s.sayHello();
        s.sayHello("Vila");

    }

    public void sayHello() {
        System.out.println("Hello");
    }

    public void sayHello(String name) {
        System.out.println("Hello" + " " + name);
    }
}
```

### 重写 Override

- 从字面上看，重写就是 重新写一遍的意思。其实就是在子类中把父类本身有的方法重新写一遍。子类继承了父类原有的方法，但有时子类并不想原封不动的继承父类中的某个方法，所以在方法名，参数列表，返回类型(除过子类中方法的返回值是父类中方法返回值的子类时)都相同的情况下，对方法体进行修改或重写，这就是重写。
- 但要注意子类函数的访问修饰权限不能少于父类的。

```java
public class Father {

    public static void main(String[] args) {
        // TODO Auto-generated method stub
        Son s = new Son();
        s.sayHello();
    }

    public void sayHello() {
        System.out.println("Hello");
    }
}

class Son extends Father{

    @Override
    public void sayHello() {
        // TODO Auto-generated method stub
        System.out.println("hello by ");
    }

}
```

### 小结

|Overide|Overload|
|-------|--------|
|发生在父类与子类之间|一个类中多态性的一种表现|
|方法名，参数列表，返回类型（除过子类中方法的返回类型是父类中返回类型的子类）必须相同|重载要求同名方法的参数列表不同(参数类型，参数个数甚至是参数顺序)|
|访问修饰符的限制一定要大于被重写方法的访问修饰符（public>protected>default>private)|重载的时候，返回值类型可以相同也可以不相同。无法以返回型别作为重载函数的区分标准|
|重写方法一定不能抛出新的检查异常或者比被重写方法申明更加宽泛的检查型异常||

### 面试题

问：重载（Overload）和重写（Override）的区别？

答：方法的重载和重写都是实现多态的方式，区别在于前者实现的是编译时的多态性，而后者实现的是运行时的多态性。重载发生在一个类中，同名的方法如果有不同的参数列表（参数类型不同、参数个数不同或者二者都不同）则视为重载；重写发生在子类与父类之间，重写要求子类被重写方法与父类被重写方法有相同的参数列表，有兼容的返回类型，比父类被重写方法更好访问，不能比父类被重写方法声明更多的异常（里氏代换原则）。重载对返回类型没有特殊的要求，不能根据返回类型进行区分。


## null处理

pageutil pagehelper

ObjectUtil.isEmpty

ObjectUtil.isNull(e)? 0:e

idea Coverage

mark 系统权责一 归一化处理

gauss
1.不同会话不能同时查询 会、修改等，会阻塞；
2.执行事务有一条失败，就会整个回滚；可使用savepoint设置达到Oracle类似效果
3.默认空间为目录，与Oracle不一样
4.空串“”和null是否一致（不确定），Oracle是一样的，MySQL是不一样的
5.建议严格astore，使用vaccum对无用数据进行清除
6.DDL自身不会提交，未提交的DDL操作会阻塞表或分区的操作；执行DDL时不会提交当前连接中的事务