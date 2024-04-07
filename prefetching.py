import numpy as np
import random

CONTENIDO_ARCHIVO = "contenido del archivo"  # Se debe reemplazar con el contenido real del archivo


def descargar_archivo_del_almacenamiento_principal():
    # Simulación de la descarga del archivo del almacenamiento principal
    tiempo_descarga = random.uniform(1, 10)
    # Simulación del tamaño del archivo
    bytes_transferidos = len(CONTENIDO_ARCHIVO)

    return bytes_transferidos, tiempo_descarga


def enviar_archivo_al_cliente(contenido_archivo):
    # Simulación del envío del archivo al cliente
    tiempo_envio = len(contenido_archivo) / tasa_bits  # Segundos

    # Se debe implementar la lógica para enviar el archivo al cliente

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



def calcular_tasa_bits_promedio(solicitudes):
    # Lista para almacenar las tasas de bits
    tasas_bits = []

    # Recorrer todas las solicitudes
    for solicitud in solicitudes:
        # Calcular la tasa de bits
        tasa_bits = solicitud["bytes_transferidos"] / solicitud["tiempo_respuesta"]

        # Agregar la tasa de bits a la lista
        tasas_bits.append(tasa_bits)

    # Calcular la tasa de bits promedio
    tasa_bits_promedio = np.mean(tasas_bits)

    return tasa_bits_promedio

# Definición de archivos
archivos = ["arch1", "arch2", "arch3", "arch4", "arch5", "arch6", "arch7", "arch8", "arch9", "arch10", "arch11", "arch12",
            "arch13", "arch14", "arch15", "arch16", "arch17", "arch18", "arch19", "arch20", "arch21", "arch22", "arch23", "arch24",]

# Definición de clientes
clientes = ["cli1", "cli2", "cli3", "cli4", "cli5", "cli6"]

# Definición del servidor
capacidad_cache = 10

# Definición del canal
tasa_bits = 1000000

# Política de prefetching
politica_prefetching = "LFU"

# diccionario con caches de los clientes
caches = {}

for cliente in clientes:
    caches[cliente] = {}

# Solicitudes
solicitudes = []

# Simulación del sistema
for i in range(100):
  # Generación de solicitud de archivo (el server decide que enviar)
  cliente = random.choice(clientes)
  archivo = random.choice(archivos)

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
            archivo_lru = min(caches[cliente].keys(), key=lambda x: caches[cliente][x]["timestamp"])
            del caches[cliente][archivo_lru]

        caches[cliente][archivo] = {"timestamp": i}
    elif politica_prefetching == "LFU":
        # Se elimina el archivo con menor frecuencia
        if len(caches[cliente]) >= capacidad_cache:
            archivo_lfu = min(caches[cliente].keys(), key=lambda x: caches[cliente][x]["frecuencia"])
            del caches[cliente][archivo_lfu]

        caches[cliente][archivo] = {"frecuencia": i}

    # Descarga del archivo del almacenamiento principal
    bytes_transferidos, tiempo_descarga = descargar_archivo_del_almacenamiento_principal()

    # Envío del archivo al cliente
    tiempo_envio = enviar_archivo_al_cliente(CONTENIDO_ARCHIVO)

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
  for cliente in clientes:
    print(f"Cache de {cliente}: {caches[cliente]}") 

print("Simulación finalizada.")
# for cliente in clientes:
#   print(f"Cache del cliente {cliente}: {caches[cliente]}")  

# Medición del rendimiento
tiempo_respuesta_promedio = calcular_tiempo_respuesta_promedio(solicitudes)
tasa_aciertos_cache = calcular_tasa_aciertos_cache(solicitudes)
#tasa_bits_promedio = calcular_tasa_bits_promedio(solicitudes)

print(f"Tiempo de respuesta promedio: {tiempo_respuesta_promedio}")
print(f"Tasa de aciertos en la caché: {tasa_aciertos_cache}")
#print(f"Tasa de bits promedio: {tasa_bits_promedio}")

