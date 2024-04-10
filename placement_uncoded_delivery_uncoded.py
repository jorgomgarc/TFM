import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pdb

#########################################################
# Function definition
#########################################################

def generar_popularidad_random():
    # Se podría cambiar y utilizar otra distribución: la normal o la que sea
    # Uniform popularity distribution
    return random.randint(1, 10)

def rellenar_archivo(archivo, size):
    letras = list("abcdefghijklmnñopqrstuvwxyz")
    random.shuffle(letras)
    with open(archivo, "wb") as f:
        for _ in range(size):
            if letras:  # Check if letras is not empty before popping
                letra = random.choice(letras).encode("utf-8")
                f.write(letra)

def descargar_archivo_del_almacenamiento_principal():

    # Calcular el tiempo de descarga
    tiempo_descarga = size / tasa_bits

    # Agregar latencia y variabilidad
    tiempo_descarga += random.uniform(0.1, 0.5)  # Segundos

    return tiempo_descarga

def timepo_envio_archivo():
    # Simulación del envío del archivo al cliente
    tiempo_envio = size / tasa_bits  # Segundos
    return tiempo_envio

def calcular_tiempo_respuesta_promedio(solicitudes):
    # Lista para almacenar los tiempos de respuesta
    tiempos_respuesta = []

    # Recorrer todas las solicitudes
    for solicitud in solicitudes:
        # Calcular el tiempo de respuesta
        tiempo_respuesta = solicitud["tiempo_fin"] - solicitud["tiempo_inicio"]

        # Agregar el tiempo de respuesta a la lista
        tiempos_respuesta.append(tiempo_respuesta)

    # Calcular el tiempo de respuesta promedio
    tiempo_respuesta_promedio = np.mean(tiempos_respuesta)

    return tiempo_respuesta_promedio

def calcular_tasa_aciertos_cache(solicitudes):
    # Número total de solicitudes
    numero_solicitudes = len(solicitudes)
    print(f"Número de solicitudes: {numero_solicitudes}")
    # Número de solicitudes que fueron hits en la caché
    numero_aciertos_cache = 0

    # Recorrer todas las solicitudes
    for solicitud in solicitudes:
    # Si la solicitud fue un hit en la caché
        if solicitud["hit_cache"]:
            # Incrementar el número de aciertos en la caché
            numero_aciertos_cache += 1
    print("Número de aciertos en la caché: ", numero_aciertos_cache)
    # Calcular la tasa de aciertos en la caché
    tasa_aciertos_cache = numero_aciertos_cache / numero_solicitudes

    # Plotear la tasa de aciertos en la caché
    # plt.figure()
    # plt.plot(numero_aciertos_cache)
    # plt.xlabel("Solicitudes")
    # plt.ylabel("Tasa de aciertos en la caché")
    # plt.title("Tasa de aciertos en la caché")
    # plt.show()

    return tasa_aciertos_cache

##########################################################
# Parameters definition
##########################################################

# Definición de archivos
archivos = [f"arch{i}" for i in range(1, 21)]

# Tamaño de los archivos
size = 1000  # 10 bytes

# Rellenar los archivos
for archivo in archivos:
    rellenar_archivo(archivo, size)

# Asignar popularidad a los archivos (de forma random)
archivos_por_popularidad = {}
popularidad_por_archivo = {}
popularidades = []
for archivo in archivos:
    popularidad = generar_popularidad_random()
    popularidades.append(popularidad)
    popularidad_por_archivo[archivo] = popularidad
    if popularidad not in archivos_por_popularidad:
        archivos_por_popularidad[popularidad] = []
    archivos_por_popularidad[popularidad].append(archivo)

# Plotear la distribución de popularidad por archivos
plt.figure()
for archivo, popularidad in popularidad_por_archivo.items():
    plt.bar(archivo, popularidad)
plt.xlabel("Archivos")
plt.ylabel("Popularidad")
plt.title("Popularidad de archivos")
plt.show(block=False)

# Definición de clientes
clientes = [f"cli{i}" for i in range(1, 7)]

# Definición del servidor
capacidad_cache = 10

# Definición del canal
tasa_bits = 1000000 # 1 Mbps

# Política de prefetching
politica_prefetching = "LFU"  # "LRU", "LFU", "HPF
frecuencia = {}

# diccionario con caches de los clientes
caches = {}

for cliente in clientes:
    caches[cliente] = {}
    frecuencia[cliente] = {}

# Solicitudes
solicitudes = []

##########################################################
# Placement phase
##########################################################

for cliente, cache in caches.items():
    # Seleccionar los archivos a almacenar en la caché
    # Implemento HPF
    archivos_cliente = sorted(archivos, key=lambda x: popularidad_por_archivo[x], reverse=True)[:capacidad_cache]

    # Almacenar los archivos en la caché
    for archivo in archivos_cliente:
        caches[cliente][archivo] = {"timestamp": 0, "popularidad": popularidad_por_archivo[archivo]}


# Plotear los archivos en cada cache de cada cliente ANTES de delivery phase
for cliente, cache in caches.items():
    archivos_cache_cliente = list(cache.keys())
    plt.figure()
    popularidades = [popularidad_por_archivo[archivo] for archivo in archivos_cache_cliente]
    plt.bar(archivos_cache_cliente, popularidades)
    plt.xlabel("Archivos")
    plt.ylabel("Popularidad")
    plt.title(f"Archivos en la caché de {cliente}")
    plt.show(block=False)

##########################################################
# Delivery phase
##########################################################

# Simulación del sistema
for i in range(100000):
    # Generación de solicitud de archivo (el server decide que enviar)
    cliente = random.choice(clientes)
    archivo = random.choice(archivos)
    
    # Inicio de la solicitud
    tiempo_inicio = i
    # Si el archivo ya está en el diccionario, incrementa su frecuencia
    if archivo in frecuencia[cliente]:
        frecuencia[cliente][archivo] += 1
    # Si el archivo no está en el diccionario, añádelo con frecuencia 1
    else:
        frecuencia[cliente][archivo] = 1
    
    # Verificación si el archivo está en la caché
    if archivo in caches[cliente]:
        # Hit en la caché
        print(f"#{i+1}: Hit para {archivo} por {cliente}")
        caches[cliente][archivo]["timestamp"] = i
        
        # Fin de la solicitud
        tiempo_fin = i
        hit_cache = True
        bytes_transferidos = size

    else:
        # Fallo en la caché
        print(f"#{i+1}: Fallo para {archivo} por {cliente}")

        # Prefetch de archivos
        if politica_prefetching == "LRU":
            # Se elimina el archivo que menos lleva en la caché
            if len(caches[cliente]) >= capacidad_cache:
                archivo_lru = min(caches[cliente].keys(), key=lambda x: caches[cliente][x]["timestamp"] * popularidad_por_archivo[x])
                del caches[cliente][archivo_lru]

            caches[cliente][archivo] = {"timestamp": i, "popularidad": popularidad_por_archivo[archivo]}

        elif politica_prefetching == "LFU":
            # Se elimina el archivo con menor frecuencia
            if len(caches[cliente]) >= capacidad_cache:
                archivo_lfu = min(caches[cliente].keys(), key=lambda x: caches[cliente][x].get("frecuencia", 0))
                del caches[cliente][archivo_lfu]

            caches[cliente][archivo] = {"frecuencia": frecuencia[cliente][archivo]}

        elif politica_prefetching == "HPF":
            # Prefetch del archivo más popular
            
            # Si la caché está llena, eliminar el archivo menos popular
            if len(caches[cliente]) >= capacidad_cache:
                archivo_menos_popular = min(caches[cliente].keys(), key=lambda x: caches[cliente][x]["popularidad"])
                del caches[cliente][archivo_menos_popular]

            # Agregar el archivo a la caché y establecer su popularidad
            caches[cliente][archivo] = {"timestamp": i, "popularidad": popularidad_por_archivo[archivo]}


        # Descarga del archivo del almacenamiento principal
        tiempo_descarga = descargar_archivo_del_almacenamiento_principal()

        # Envío del archivo al cliente
        tiempo_envio = timepo_envio_archivo()

        # Fin de la solicitud
        tiempo_fin = i + tiempo_descarga + tiempo_envio
        hit_cache = False
        bytes_transferidos = size

    # Almacenamiento de información de la solicitud
    solicitudes.append({
    "tiempo_inicio": tiempo_inicio,
    "tiempo_fin": tiempo_fin,
    "hit_cache": hit_cache,
    "bytes_transferidos": bytes_transferidos,
  })
  

print("Simulación finalizada.")

# Medición del rendimiento
tiempo_respuesta_promedio = calcular_tiempo_respuesta_promedio(solicitudes)
tasa_aciertos_cache = calcular_tasa_aciertos_cache(solicitudes)

print(f"Tiempo de respuesta promedio: {tiempo_respuesta_promedio}")
print(f"Tasa de aciertos en la caché: {tasa_aciertos_cache}")

if politica_prefetching == "HPF":
# Plotear los archivos en cada cache de cada cliente y su nivel de popularidad
    for cliente, cache in caches.items():
        archivos = list(cache.keys())
        plt.figure()
        popularidades = [popularidad_por_archivo[archivo] for archivo in archivos]
        plt.bar(archivos, popularidades)
        plt.xlabel("Archivos")
        plt.ylabel("Popularidad")
        plt.title(f"Archivos en la caché de {cliente}")
        plt.show(block=False)
elif politica_prefetching == "LFU":
# Plotear los archivos en cada cache de cada cliente y su frecuencia
    for cliente, cache in caches.items():
        archivos_cache_cliente = list(cache.keys())
        frecuencias = [frecuencia[cliente][archivo] for archivo in archivos_cache_cliente]
        plt.figure()
        plt.bar(archivos_cache_cliente, frecuencias)
        plt.xlabel("Archivos")
        plt.ylabel("Frecuencia")
        plt.title(f"Frecuencia de archivos en la caché de {cliente}")
        plt.show(block=False)


plt.show()