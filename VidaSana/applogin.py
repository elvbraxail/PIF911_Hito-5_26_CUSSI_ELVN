from flask import Flask,request,jsonify,render_template,session,redirect
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash,check_password_hash
from functools import wraps
import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

app=Flask(__name__)
app.secret_key="vidasana123"

CORS(app)

client=MongoClient("mongodb://localhost:27017/")
db=client["sistema_emociones"]

usuarios=db["usuarios"]
pacientes=db["pacientes"]
personal=db["personal"]
evaluaciones=db["evaluaciones"]
analisis_ia=db["analisis_ia"]

# =====================================
# MODELO IA
# =====================================

MODELO_PATH="modelo_bert_ansiedad_social"

import os

MODELO_PATH = r"C:\Users\Elvin\Desktop\modelo_bert_ansiedad_social"

print("\nMODELO:")
print(MODELO_PATH)

print("\nEXISTE:")
print(os.path.exists(MODELO_PATH))

if os.path.exists(MODELO_PATH):
    print("\nARCHIVOS:")
    print(os.listdir(MODELO_PATH))

tokenizer=AutoTokenizer.from_pretrained(
    MODELO_PATH
)

model=AutoModelForSequenceClassification.from_pretrained(
    MODELO_PATH
)

CLASES=[

    "angustia_panico",

    "frustracion_impotencia",

    "miedo_intenso"

]


def login_requerido(f):
    @wraps(f)
    def decorador(*args,**kwargs):

        if "usuario_id" not in session:
            return redirect("/")

        return f(*args,**kwargs)

    return decorador


def rol_requerido(*roles):

    def wrapper(f):

        @wraps(f)
        def decorador(*args,**kwargs):

            if "usuario_id" not in session:
                return redirect("/")

            if session.get("rol") not in roles:
                return "Acceso denegado",403

            return f(*args,**kwargs)

        return decorador
    return wrapper

def analizar_emocion(texto):

    inputs=tokenizer(

        texto,

        return_tensors="pt",

        truncation=True,

        padding=True,

        max_length=256

    )

    with torch.no_grad():

        outputs=model(**inputs)

    probs=torch.softmax(
        outputs.logits,
        dim=1
    ).numpy()[0]

    indice=np.argmax(probs)

    emocion=CLASES[indice]

    confianza=float(
        probs[indice]
    )

    return emocion,confianza,probs

@app.route(
    "/analizar_emociones",
    methods=["POST"]
)
@login_requerido
def analizar_emociones():

    datos=request.get_json()

    texto=datos["texto"]

    emocion,confianza,probs=analizar_emocion(
        texto
    )

    if confianza>=0.85:

        nivel="ALTO"

    elif confianza>=0.70:

        nivel="MODERADO"

    else:

        nivel="BAJO"

    interpretacion=f"""

    Emoción detectada:
    {emocion}

    Intensidad:
    {round(confianza*100,2)}%

    Nivel:
    {nivel}

    """

    return jsonify({

        "emocion":emocion,

        "intensidad":
        round(
            confianza*100,
            2
        ),

        "interpretacion":
        interpretacion

    })

#=====================================
# VISTAS
#=====================================

@app.route("/")
def login_page():
    return render_template("login.html")


@app.route("/sistema")
@login_requerido
def sistema():
    return render_template("sistema.html")


@app.route("/pacientes")
@rol_requerido("admin","psicologo")
def vista_pacientes():
    return render_template("pacientes.html")


@app.route("/personal")
@rol_requerido("admin","psicologo")
def vista_personal():
    return render_template("personal.html")


@app.route("/registro")
@rol_requerido("admin","psicologo")
def vista_registro():
    return render_template("registro.html")


@app.route("/ayuda")
@login_requerido
def vista_ayuda():
    return render_template("ayuda.html")


@app.route("/ubicacion")
@login_requerido
def vista_ubicacion():
    return render_template("ubicacion.html")



#=====================================
# LOGIN
#=====================================

@app.route("/login",methods=["POST"])
def login():

    data=request.get_json()

    usuario=data.get("usuario")
    password=data.get("password")

    user=usuarios.find_one({
        "usuario":usuario
    })

    if user and check_password_hash(
        user["password"],
        password
    ):

        session["usuario_id"]=str(user["_id"])
        session["usuario"]=user["usuario"]
        session["rol"]=user["rol"]

        return jsonify({
            "status":"ok",
            "rol":user["rol"]
        })

    return jsonify({
        "status":"error"
    }),401



@app.route("/sesion")
def ver_sesion():

    if "usuario_id" in session:

        return jsonify({
            "usuario":session["usuario"],
            "rol":session["rol"]
        })

    return jsonify({
        "error":"No hay sesion"
    }),401



@app.route("/logout")
def logout():

    session.clear()

    return jsonify({
        "status":"sesion cerrada"
    })



#=====================================
# TEST
#=====================================

@app.route("/guardar_test",methods=["POST"])
@login_requerido
def guardar_test():

    data=request.get_json()

    nuevo={

        "usuario_id":session["usuario_id"],
        "fecha":datetime.now(),
        "puntaje_total":data["puntaje"],
        "nivel":data["nivel"],
        "respuestas":data["respuestas"]
    }

    evaluaciones.insert_one(nuevo)

    return jsonify({
        "status":"guardado"
    })



@app.route("/mis_resultados")
@login_requerido
def mis_resultados():

    datos=list(
        evaluaciones.find(
            {
                "usuario_id":session["usuario_id"]
            },
            {
                "_id":0
            }
        )
    )

    return jsonify(datos)



@app.route("/registrar_paciente",methods=["POST"])
@rol_requerido("admin","psicologo")
def registrar_paciente():

    data=request.get_json()

    if usuarios.find_one({"usuario":data["usuario"]}):

        return jsonify({
            "status":"error",
            "mensaje":"Usuario ya existe"
        }),400

    nuevo_usuario={

        "usuario":data["usuario"],
        "password":generate_password_hash(
            data["password"]
        ),
        "rol":"paciente",

        "nombre":data["nombre"],
        "ci":data["ci"],
        "edad":data["edad"],
        "telefono":data["telefono"],
        "correo":data["correo"],
        "sexo":data["sexo"],
        "motivo":data["motivo"]
    }

    r=usuarios.insert_one(nuevo_usuario)

    pacientes.insert_one({

        "usuario_id":str(r.inserted_id),

        "nombre":data["nombre"],
        "ci":data["ci"],
        "edad":data["edad"],
        "telefono":data["telefono"],
        "correo":data["correo"],
        "sexo":data["sexo"],
        "motivo":data["motivo"]
    })


    return jsonify({
        "status":"ok"
    })



#=====================================
# REGISTRAR PSICOLOGO
#=====================================

@app.route("/registrar_psicologo",methods=["POST"])
@rol_requerido("admin","psicologo")
def registrar_psicologo():

    data=request.get_json()

    if usuarios.find_one({"usuario":data["usuario"]}):

        return jsonify({
            "status":"error",
            "mensaje":"Usuario existe"
        })


    nuevo_usuario={

        "usuario":data["usuario"],
        "password":generate_password_hash(
            data["password"]
        ),

        "rol":"psicologo",

        "nombre":data["nombre"],
        "ci":data["ci"],
        "matricula":data["matricula"],
        "especialidad":data["especialidad"],
        "telefono":data["telefono"],
        "correo":data["correo"]
    }

    r=usuarios.insert_one(
        nuevo_usuario
    )


    personal.insert_one({

        "usuario_id":str(
            r.inserted_id
        ),

        "nombre":data["nombre"],
        "correo":data["correo"],
        "telefono":data["telefono"],

        "cargo":"Psicólogo",
        "especialidad":data["especialidad"]
    })

    return jsonify({
        "status":"ok"
    })



#=====================================
# PERSONAL CRUD
#=====================================

@app.route("/api/personal",methods=["GET"])
@rol_requerido("admin","psicologo")
def get_personal():

    lista=list(
        personal.find()
    )

    for p in lista:
        p["_id"]=str(
            p["_id"]
        )

    return jsonify(lista)



@app.route("/api/personal",methods=["POST"])
@rol_requerido("admin","psicologo")
def add_personal():

    data=request.get_json()

    personal.insert_one({

        "nombre":data["nombre"],
        "correo":data["correo"],
        "telefono":data["telefono"],
        "cargo":data["cargo"],
        "especialidad":data.get(
            "especialidad",""
        )
    })

    return jsonify({
        "status":"creado"
    })



#=====================================
# EDITAR PERSONAL (ARREGLADO)
#=====================================

@app.route("/api/personal/<id>",methods=["PUT"])
@rol_requerido("admin","psicologo")
def editar_personal(id):

    data=request.get_json()

    nuevos_datos={

        "nombre":data.get("nombre"),
        "correo":data.get("correo"),
        "telefono":data.get("telefono"),
        "cargo":data.get("cargo"),
        "especialidad":data.get(
            "especialidad"
        )
    }


    resultado=personal.update_one(
        {
            "_id":ObjectId(id)
        },
        {
            "$set":nuevos_datos
        }
    )


    # sincroniza con usuarios si existe usuario_id asociado
    registro=personal.find_one({
        "_id":ObjectId(id)
    })

    if registro and "usuario_id" in registro:

        try:
            usuarios.update_one(
                {
                    "_id":ObjectId(
                        registro["usuario_id"]
                    )
                },
                {
                    "$set":{
                        "nombre":data.get("nombre"),
                        "correo":data.get("correo"),
                        "telefono":data.get("telefono"),
                        "especialidad":data.get(
                           "especialidad"
                        )
                    }
                }
            )
        except:
            pass


    if resultado.modified_count>0:

        return jsonify({
            "status":"actualizado"
        })

    return jsonify({
        "status":"sin cambios"
    })



#=====================================
# ELIMINAR PERSONAL
#=====================================

@app.route("/api/personal/<id>",methods=["DELETE"])
@rol_requerido("admin","psicologo")
def eliminar_personal(id):

    doc=personal.find_one({
        "_id":ObjectId(id)
    })

    if doc and "usuario_id" in doc:
        try:
            usuarios.delete_one({
                "_id":ObjectId(
                    doc["usuario_id"]
                )
            })
        except:
            pass


    personal.delete_one({
        "_id":ObjectId(id)
    })

    return jsonify({
        "status":"eliminado"
    })



#=====================================
# PACIENTES CRUD
#=====================================

@app.route("/api/pacientes",methods=["GET"])
@rol_requerido("admin","psicologo")
def get_pacientes():

    lista=list(
        pacientes.find()
    )

    for p in lista:
        p["_id"]=str(p["_id"])

    return jsonify(lista)



@app.route("/api/pacientes/<id>",methods=["DELETE"])
@rol_requerido("admin","psicologo")
def borrar_paciente(id):

    pacientes.delete_one({
        "_id":ObjectId(id)
    })

    return jsonify({
        "status":"eliminado"
    })

#=====================================
# ACTUALIZAR PACIENTE (AGREGAR ESTO)
#=====================================

@app.route("/api/pacientes",methods=["POST"])
@rol_requerido("admin","psicologo")
def add_paciente():

    data=request.get_json()

    data["fecha_actualizacion"]=datetime.now()

    pacientes.insert_one(data)

    return jsonify({
        "status":"creado"
    })



@app.route("/api/pacientes/<id>",methods=["PUT"])
@rol_requerido("admin","psicologo")
def update_paciente(id):

    data=request.get_json()

    data["fecha_actualizacion"]=datetime.now()

    resultado=pacientes.update_one(
        {
            "_id":ObjectId(id)
        },
        {
            "$set":data
        }
    )


    # sincroniza también si paciente viene de usuarios
    paciente_doc=pacientes.find_one({
        "_id":ObjectId(id)
    })

    if paciente_doc and "usuario_id" in paciente_doc:

        try:
            usuarios.update_one(
                {
                    "_id":ObjectId(
                        paciente_doc["usuario_id"]
                    )
                },
                {
                    "$set":{
                        "nombre":data.get("nombre"),
                        "ci":data.get("ci"),
                        "edad":data.get("edad"),
                        "telefono":data.get("telefono"),
                        "correo":data.get("correo"),
                        "sexo":data.get("sexo"),
                        "motivo":data.get("motivo")
                    }
                }
            )
        except:
            pass


    if resultado.modified_count>0:

        return jsonify({
            "status":"actualizado"
        })

    return jsonify({
        "status":"sin cambios"
    })

#=====================================
# RUN
#=====================================

if __name__=="__main__":
    app.run(debug=True)