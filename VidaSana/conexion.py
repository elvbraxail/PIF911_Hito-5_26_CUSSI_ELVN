from pymongo import MongoClient

try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    
    client.server_info()

    db = client["base_de_datos"]

    print("Conectado a MongoDB")
    print(client.list_database_names())

except Exception as e:
    print("Error:", e)