import random
import matplotlib.pyplot as plt
import utils

##########################################################
# Parameters definition
##########################################################

# Definición de archivos
archivos = [f"file_{i}" for i in range(1, 21)]

# Tamaño de los archivos
size = 1000  # 10 bytes

# Definición de clientes
clientes = [f"cli_{i}" for i in range(1, 31)]

# Definición del servidor
capacidad_cache = 10

# Definición del canal
tasa_bits = 1000000  # 1 Mbps

# Rellenar los archivos
for archivo in archivos:
    utils.rellenar_archivo(archivo, size)

# Asignar popularidad a los archivos (de forma random)
archivos_por_popularidad = {}
popularidad_por_archivo = {}
popularidades = []

for i, archivo in enumerate(archivos, start=1):
    popularidad = i
    popularidades.append(popularidad)
    popularidad_por_archivo[archivo] = popularidad
    if popularidad not in archivos_por_popularidad:
        archivos_por_popularidad[popularidad] = []
    archivos_por_popularidad[popularidad].append(archivo)



# Diccionario con caches de los clientes
caches = {}

for cliente in clientes:
    caches[cliente] = {}

# Solicitudes
solicitudes = []

##########################################################
# Placement phase
##########################################################

prefetching_policy = "popularity"  # "random", "popularity"
num_solicitudes = 10000
politica_delivery = "random"  # "random", "popularidad"

for cliente, cache in caches.items():
    if prefetching_policy == "popularity":
                archivos_cliente = sorted(archivos, key=lambda x: popularidad_por_archivo[x], reverse=True)[:capacidad_cache]
    elif prefetching_policy == "random":
        archivos_cliente = random.sample(archivos, min(len(archivos), capacidad_cache))  # Fill cache with random files

    # Almacenar los archivos en la caché
    for archivo in archivos_cliente:
        caches[cliente][archivo] = {"timestamp": 0, "popularidad": popularidad_por_archivo[archivo]}

##########################################################
# Delivery phase
##########################################################

cache_hits = 0
satisfechos = 0
repetidos = 0
cache_hits_list = []

# Simulación del sistema
for i in range(num_solicitudes):  # nº solicitudes
    # Resetear las variables
    peticiones = []
    cache_hits = 0
    # Generación de peticiones de archivo por cada cliente
    for cliente in clientes:
        if politica_delivery == "random":
            archivo = random.choice(archivos)  # Selecciono un archivo aleatorio
        elif politica_delivery == "popularidad":
            archivo = random.choices(archivos, weights=[popularidad_por_archivo[x] for x in archivos])[0]  # Selecciono un archivo con mayor probabilidad basado en su popularidad
        # archivo = max(archivos, key=lambda x: popularidad_por_archivo[x])  # Selecciono el archivo más popular
        peticion = {"archivo": archivo, "cliente": cliente}
        # Compruebo si el cliente tiene cacheado dicho archivo
        if caches[cliente].get(archivo) is None:
            peticiones.append(peticion)
        else:
            # Hit en la caché
            caches[cliente][archivo]["timestamp"] = i
            tiempo_inicio = i

            # Fin de la solicitud
            tiempo_fin = i
            hit_cache = True
            cache_hits += 1
            bytes_transferidos = 0
            # Almacenamiento de información de la solicitud
            solicitudes.append({
                "tiempo_inicio": tiempo_inicio,
                "tiempo_fin": tiempo_fin,
                "hit_cache": hit_cache,
                "bytes_transferidos": bytes_transferidos,
            })
        cache_hits_list.append(cache_hits)

    # Compruebo si hay peticiones de archivos repetidos, para enviarlos en un solo mensaje
    archivos_a_enviar, repes = utils.comprobar_peticiones_repetidas(peticiones)
    satisfechos += len(archivos_a_enviar)
    repetidos += repes
    tiempo_inicio = i
    # Enviar mensajes a los clientes para entregar los archivos solicitados
    # Calcular el tiempo en enviar los archivos a los clientes
    tiempo_envio = utils.tiempo_envio_archivos(archivos_a_enviar, tasa_bits, size)

    # Actualizar las caches de los clientes:
    for peticion in peticiones:
        archivo = peticion["archivo"]
        cliente = peticion["cliente"]
        # Actualización de la cache
        # Si la caché está llena, eliminar el archivo menos popular
        if len(caches[cliente]) >= capacidad_cache:
            archivo_menos_popular = min(caches[cliente].keys(), key=lambda x, cliente=cliente: caches[cliente][x]["popularidad"])
            del caches[cliente][archivo_menos_popular]

        # Agregar el archivo a la caché y establecer su popularidad
        caches[cliente][archivo] = {"timestamp": i, "popularidad": popularidad_por_archivo[archivo]}

        # Descarga del archivo del almacenamiento principal
        tiempo_descarga = utils.descargar_archivo_del_almacenamiento_principal(size, tasa_bits)

        # Fin de la solicitud
        tiempo_fin = i + tiempo_descarga + tiempo_envio
        hit_cache = False
        bytes_transferidos = size * len(archivos_a_enviar)

        # Almacenamiento de información de la solicitud
        solicitudes.append({
            "tiempo_inicio": tiempo_inicio,
            "tiempo_fin": tiempo_fin,
            "hit_cache": hit_cache,
            "bytes_transferidos": bytes_transferidos,
        })

#############################################################
# Plotear y printear resultados
#############################################################

print("Simulation completed.")

# Medición del rendimiento
tiempo_respuesta_promedio = utils.calcular_tiempo_respuesta_promedio(solicitudes)
tasa_aciertos_cache = utils.calcular_tasa_aciertos_cache(solicitudes)
ancho_de_banda = utils.calcular_ancho_de_banda(solicitudes)

print(f"Average response time: {tiempo_respuesta_promedio:.3f}")
print(f"Cache hit rate: {tasa_aciertos_cache:.3f}")
print(f"Satisfied Requests: {satisfechos}")
print(f"Repeated Requests: {repetidos}")
print(f"Bandwidth: {ancho_de_banda:.3f} bits/second")

# Plot the cache hits
plt.figure()
plt.plot(range(num_solicitudes * len(clientes)), cache_hits_list)
plt.xlabel('Number of requests')
plt.ylabel('Hits in the cache')
plt.title('Hits in the cache vs. Number of requests')
plt.show()

# Plotear los archivos en cada cache de cada cliente y su nivel de popularidad
# for cliente, cache in caches.items():
#     archivos = list(cache.keys())
#     plt.figure()
#     popularidades = [popularidad_por_archivo[archivo] for archivo in archivos]
#     plt.bar(archivos, popularidades)
#     plt.xlabel("Archivos")
#     plt.ylabel("Popularidad")
#     plt.title(f"Archivos en la caché de {cliente} después de la fase de entrega")

# plt.show()

# # Suponiendo que ya tienes los cálculos para ancho_de_banda y tiempo_respuesta_promedio

# # Datos para plotear
# tiempos = list(range(len(solicitudes)))
# ancho_de_banda_list = [solicitud['bytes_transferidos'] * 8 / (solicitud['tiempo_fin'] - solicitud['tiempo_inicio'] + 1) for solicitud in solicitudes]
# tiempo_respuesta_promedio_list = [solicitud['tiempo_fin'] - solicitud['tiempo_inicio'] for solicitud in solicitudes]

# # Promedio acumulado de ancho de banda
# ancho_de_banda_acumulado = []
# suma_ancho_de_banda = 0
# for i in range(len(ancho_de_banda_list)):
#     suma_ancho_de_banda += ancho_de_banda_list[i]
#     ancho_de_banda_acumulado.append(suma_ancho_de_banda / (i + 1))

# # Promedio acumulado de tiempo de respuesta
# tiempo_respuesta_promedio_acumulado = []
# suma_tiempo_respuesta = 0
# for i in range(len(tiempo_respuesta_promedio_list)):
#     suma_tiempo_respuesta += tiempo_respuesta_promedio_list[i]
#     tiempo_respuesta_promedio_acumulado.append(suma_tiempo_respuesta / (i + 1))

# # Crear el gráfico
# fig, ax1 = plt.subplots()

# # Plotear el tiempo de respuesta promedio
# ax1.set_xlabel('Number of requests')
# ax1.set_ylabel('Average Response Time', color='tab:blue')
# ax1.plot(tiempos, tiempo_respuesta_promedio_acumulado, color='tab:blue')
# ax1.tick_params(axis='y', labelcolor='tab:blue')

# # Crear un segundo eje para el ancho de banda
# ax2 = ax1.twinx()
# ax2.set_ylabel('Bandwidth (bits/second)', color='tab:orange')
# ax2.plot(tiempos, ancho_de_banda_acumulado, color='tab:orange')
# ax2.tick_params(axis='y', labelcolor='tab:orange')

# # Título y mostrar el gráfico
# plt.title('Average Response Time and Bandwidth Over Time')
# fig.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Datos simulados para cada configuración
# Configuración 1: prefetching_policy=random, politica_delivery=random
avg_response_time_1 = 0.1  # Ejemplo de valor
bandwidth_1 = 1.15e6  # Ejemplo de valor

# Configuración 2: prefetching_policy=random, politica_delivery=popularity
avg_response_time_2 = 0.09  # Ejemplo de valor
bandwidth_2 = 1.25e6  # Ejemplo de valor

# Configuración 3: prefetching_policy=popularity, politica_delivery=random
avg_response_time_3 = 0.08  # Ejemplo de valor
bandwidth_3 = 1.2e6  # Ejemplo de valor

# Configuración 4: prefetching_policy=popularity, politica_delivery=popularity
avg_response_time_4 = 0.06  # Ejemplo de valor
bandwidth_4 = 0.9e6  # Ejemplo de valor

# Crear una lista de los tiempos de respuesta y anchos de banda para las 4 configuraciones
avg_response_times = [avg_response_time_1, avg_response_time_2, avg_response_time_3, avg_response_time_4]
bandwidths = [bandwidth_1, bandwidth_2, bandwidth_3, bandwidth_4]

# Etiquetas para las configuraciones
labels = ['Rand/Rand', 'Rand/Pop', 'Pop/Rand', 'Pop/Pop']

# Posiciones de las barras
x = np.arange(len(labels))

# Ancho de las barras
width = 0.4

# Crear el gráfico de barras para el tiempo de respuesta promedio
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.bar(x, avg_response_times, width, color='b')
ax1.set_xlabel('Configurations')
ax1.set_ylabel('Average Response Time (s)')
ax1.set_title('Average Response Time for Different Configurations')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha='right')

# Crear el gráfico de barras para el ancho de banda
ax2.bar(x, bandwidths, width, color='orange')
ax2.set_xlabel('Configurations')
ax2.set_ylabel('Bandwidth (bits/second)')
ax2.set_title('Bandwidth for Different Configurations')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right')

# Ajustar el diseño y mostrar el gráfico
fig.tight_layout()
plt.show()
