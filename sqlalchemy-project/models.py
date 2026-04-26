from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float
#zadanie 12
from sqlalchemy import Date

Base = declarative_base()

class Product(Base):
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    category = Column(String(100))
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)
    description = Column(String(500))

#zadanie 12
class Article(Base):
    __tablename__ = 'articles'
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    content = Column(String)
    published_date = Column(Date)
