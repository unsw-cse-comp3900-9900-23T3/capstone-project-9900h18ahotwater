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

    def find_by_email(self, email):
        return dbsession.query(User).filter(User.email == email).first()

    def find_by_phone(self, phone):
        return dbsession.query(User).filter(User.phone == phone).first()


class History(Base):
    __table__ = Table('history', md, autoload=True)

    def find_by_user_id(self, user_id):
        return dbsession.query(History).filter(History.user_id == user_id).all()

    def find_by_user_id_and_data_id(self, user_id, data_id):
        return dbsession.query(History).filter(and_(History.user_id == user_id, History.data_id == data_id)).first()


class Data(Base):
    __table__ = Table('data', md, autoload=True)

    def find_by_data_id(self, data_id):
        return dbsession.query(Data).filter(Data.data_id == data_id).first()

    def find_data(self,img1, img2, text):
        return dbsession.query(Data).filter(and_(Data.img1 == img1, Data.img2 == img2, Data.text == text)).first()
# if __name__ == "__main__":
#     res = dbsession.query(User).all()
#     for i in res:
#         print(i.user_id, i.nickname, i.username, i.password, i.email, i.phone)
#     user = User(nickname="123@a.com", email="123@a.com", phone="12345", password="asdfqw")
#     dbsession.add(user)
#     dbsession.commit()

