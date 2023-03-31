import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234"
)
mycursor = mydb.cursor()
mycursor.execute("USE mlops")

'''
mycursor.execute("CREATE DATABASE mlops")

mycursor.execute("SHOW DATABASES")
res = mycursor.fetchall()
print(res)


mycursor.execute("USE mlops")
mycursor.execute(
    """
    CREATE TABLE users(
        name VARCHAR(100),
        email VARCHAR(100),
        password VARCHAR(100)
    )
    """)
'''

def newuser(email,uname,psw):
    mycursor.execute(f"SELECT * FROM users WHERE name='{uname}' or email='{email}'")
    if mycursor.fetchall():
        print("user already exists")
        return False
        
    mycursor.execute(
        "INSERT INTO users (email,name,password) VALUES (%s,%s,%s)",
        (email,uname,psw)
    )
    mydb.commit()
    return True

def loginuser(uname,psw):
    mycursor.execute(f"SELECT password FROM users WHERE name='{uname}'")
    p = mycursor.fetchone()
    if p and psw==p[0]:
        print("User verified!")
        return True
    return False