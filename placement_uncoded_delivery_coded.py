import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pdb
import utils

CONTENIDO_ARCHIVO = "contenido del archivo"  # Se debe reemplazar con el contenido real del archivo

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

def dividir_archivo(archivo, numero_partes):
    partes = []
    for i in range(numero_partes):
        contenido = CONTENIDO_ARCHIVO
        parte = contenido[i::numero_partes]
        partes.append(parte)
    return partes

def tiempo_envio_archivo():
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

    # Número de solicitudes que fueron hits en la caché
    numero_aciertos_cache = 0

    # Recorrer todas las solicitudes
    for solicitud in solicitudes:
    # Si la solicitud fue un hit en la caché
        if solicitud["hit_cache"]:
            # Incrementar el número de aciertos en la caché
            numero_aciertos_cache += 1

    # Calcular la tasa de aciertos en la caché
    tasa_aciertos_cache = numero_aciertos_cache / numero_solicitudes

    return tasa_aciertos_cache

def generate_coded_packets(server_files, user_requests, cache_sizes):
    """
    Genera paquetes codificados usando XOR para la entrega de la fase de caché codificada.

    Args:
        server_files: Lista de listas, donde cada lista interna representa partes de un archivo.
        user_requests: Diccionario, donde las claves son usuarios y los valores son números de partes solicitadas.
        cache_sizes: Lista de enteros que representan los tamaños de caché para cada usuario.

    Returns:
        Diccionario, donde las claves son usuarios y los valores son listas de paquetes codificados.
    """

    coded_packets = {}
    for user, requested_parts in user_requests.items():
        cached_parts = random.sample(server_files, cache_sizes[user])
        coded_packets[user] = []
        for requested_part in requested_parts:
            packet = cached_parts[0] ^ cached_parts[1]
            for i in range(2, len(cached_parts)):
                packet ^= cached_parts[i]
                coded_packets[user].append(packet)

    return coded_packets

def decode_parts(cache_sizes, user_requests, coded_packets):
    """
    Permite a los usuarios decodificar las partes solicitadas utilizando la caché y los paquetes recibidos.

    Args:
        cache_sizes: Lista de enteros que representan los tamaños de caché para cada usuario.
        user_requests: Diccionario, donde las claves son usuarios y los valores son números de partes solicitadas.
        coded_packets: Diccionario, donde las claves son usuarios y los valores son listas de paquetes codificados.

    Returns:
        Diccionario, donde las claves son usuarios y los valores son las partes decodificadas.
    """

    decoded_parts = {}
    for user, requested_parts in user_requests.items():
        decoded_parts[user] = []
        for i, requested_part in enumerate(requested_parts):
            cached_parts = random.sample(server_files, cache_sizes[user])
            decoded_part = cached_parts[0] ^ coded_packets[user][i]
            for j in range(1, len(cached_parts)):
                decoded_part ^= cached_parts[j]
            decoded_parts[user].append(decoded_part)

    return decoded_parts

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

# Plotear la distribución de popularidad en archivos
plt.figure()
for archivo, popularidad in archivos_por_popularidad.items():
    plt.bar(popularidad, archivo)
plt.xlabel("Archivos")
plt.ylabel("Popularidad")
plt.title("Popularidad de archivos")
plt.show(block=False)

# Dividir los archivos en partes
numero_partes = 4
partes_por_archivo = {}
for archivo in archivos:
    partes = dividir_archivo(archivo, numero_partes)
    partes_por_archivo[archivo] = [f"{archivo}_{i+1}" for i in range(numero_partes)]

    print(partes_por_archivo[archivo])

    
# Definición de clientes
clientes = [f"cli{i}" for i in range(1, 7)]

# Definición del servidor
capacidad_cache = 10

# Definición del canal
tasa_bits = 1000000 # 1 Mbps

# Política de prefetching
politica_prefetching = "HPF"  # "LRU", "LFU", "HPF

# Diccionario con caches de los clientes
caches = {}

for cliente in clientes:
    caches[cliente] = {}

# Solicitudes
solicitudes = []
# Peticiones
peticiones = {}

# Diccionario en el que se guarda el estado de las caches de los clientes
servidor = {}
for cliente in clientes:
    servidor[cliente] = {}
    for archivo in archivos:
        for parte in partes_por_archivo[archivo]:
            servidor[cliente][archivo][parte] = 0

print(servidor)
pdb.set_trace()

# Simulación del sistema
for i in range(100):
    # Generación de peticiones:
    for cliente in clientes:
        peticiones[cliente] = random.choice(archivos)
    
    # Inicio de la solicitud
    tiempo_inicio = i
    
    # Verificación si el archivo está en la caché
    if archivo in caches[cliente]:
        # Hit en la caché
        print(f"#{i+1}: Hit para {archivo} por {cliente}")

        # Fin de la solicitud
        tiempo_fin = i
        hit_cache = True
        bytes_transferidos = len(CONTENIDO_ARCHIVO)

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
                archivo_lfu = min(caches[cliente].keys(), key=lambda x: caches[cliente][x]["frecuencia"])
                del caches[cliente][archivo_lfu]

            caches[cliente][archivo] = {"frecuencia": i}

        elif politica_prefetching == "HPF":
            # Prefetch del archivo más popular
            # Si el archivo está en la caché, actualizar timestamp
            if archivo in caches[cliente]:
                caches[cliente][archivo]["timestamp"] = i
            else:
                # Si la caché está llena, eliminar el archivo menos popular
                if len(caches[cliente]) >= capacidad_cache:
                    archivo_menos_popular = min(caches[cliente].keys(), key=lambda x: caches[cliente][x]["popularidad"])
                    del caches[cliente][archivo_menos_popular]

                # Agregar el archivo a la caché y establecer su popularidad
                caches[cliente][archivo] = {"timestamp": i, "popularidad": popularidad_por_archivo[archivo]}


    # Descarga del archivo del almacenamiento principal
    bytes_transferidos, tiempo_descarga = descargar_archivo_del_almacenamiento_principal()

    # Envío del archivo al cliente
    tiempo_envio = timepo_envio_archivo()

    # Fin de la solicitud
    tiempo_fin = i + tiempo_descarga + tiempo_envio
    hit_cache = False
    bytes_transferidos = len(CONTENIDO_ARCHIVO)

    # Almacenamiento de información de la solicitud
    solicitudes.append({
    "tiempo_inicio": tiempo_inicio,
    "tiempo_fin": tiempo_fin,
    "hit_cache": hit_cache,
    "bytes_transferidos": bytes_transferidos,
    "timepo_respuesta": tiempo_fin - tiempo_inicio,
  })
  

print("Simulación finalizada.")

# Medición del rendimiento
tiempo_respuesta_promedio = calcular_tiempo_respuesta_promedio(solicitudes)
tasa_aciertos_cache = calcular_tasa_aciertos_cache(solicitudes)

print(f"Tiempo de respuesta promedio: {tiempo_respuesta_promedio}")
print(f"Tasa de aciertos en la caché: {tasa_aciertos_cache}")

# Plotear los archivos en cada cache de cada cliente y su nivel de popularidad
for cliente, cache in caches.items():
    archivos = list(cache.keys())
    plt.figure()
    popularidades = [popularidad_por_archivo[archivo] for archivo in archivos]
    plt.bar(archivos, popularidades)
    plt.xlabel("Archivos")
    plt.ylabel("Popularidad")
    plt.title(f"Archivos en la caché de {cliente}")
    plt.show()