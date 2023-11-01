from sqlalchemy import create_engine, Column, Integer, String, DateTime, Table, \
    MetaData, or_, func, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, session, scoped_session

# connect to mysql
password = 'Ren894493655' #need to change
engine = create_engine('mysql+pymysql://root:'+password+'@localhost:3306/Comp9900', echo=False, pool_size=1000)

#create session
DBsession = sessionmaker(bind=engine)
dbsession = scoped_session(DBsession)   # thread safe
Base = declarative_base()
md = MetaData(bind=engine)

class User(Base):
    __table__ = Table('user', md, autoload=True)

class History(Base):
    __table__ = Table('history', md, autoload=True)

class Data(Base):
    __table__ = Table('data', md, autoload=True)

if __name__ == "__main__":
    res = dbsession.query(User).all()
    for i in res:
        print(i.user_id, i.nickname, i.username, i.password, i.email, i.phone)

