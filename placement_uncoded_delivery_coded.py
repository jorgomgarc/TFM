import random
import matplotlib.pyplot as plt
import pdb
from pprint import pprint
import utils
import numpy as np


##########################################################
# Parameters definition
##########################################################

# Definición de archivos
archivos = [f"file_{i}" for i in range(1, 51)]

# Tamaño de los archivos
size = 1000  # 10 bytes

# Rellenar los archivos
for archivo in archivos:
    utils.rellenar_archivo(archivo, size)

# Asignar popularidad a los archivos (de forma random)
archivos_por_popularidad = {}
popularidad_por_archivo = {}
popularidades = []
for archivo in archivos:
    popularidad = utils.generar_popularidad_random()
    popularidades.append(popularidad)
    popularidad_por_archivo[archivo] = popularidad
    if popularidad not in archivos_por_popularidad:
        archivos_por_popularidad[popularidad] = []
    archivos_por_popularidad[popularidad].append(archivo)

# Plotear la distribución de popularidad en archivos
# plt.figure()
# for archivo, popularidad in archivos_por_popularidad.items():
#     plt.bar(popularidad, archivo)
# plt.xlabel("Archivos")
# plt.ylabel("Popularidad")
# plt.title("Popularidad de archivos")
# plt.show(block=False)

# Dividir los archivos en partes
# numero_partes = 4
# partes_por_archivo = {}
# for archivo in archivos:
#     partes = utils.dividir_archivo(archivo, numero_partes)
#     partes_por_archivo[archivo] = [f"{archivo}_{i+1}" for i in range(numero_partes)]

#     print(partes_por_archivo[archivo])

    
# Definición de clientes
clientes = [f"cli_{i}" for i in range(1, 11)]

# Definición del servidor
capacidad_cache = 10

# Definición del canal
tasa_bits = 1000000 # 1 Mbps

# Diccionario con caches de los clientes
caches = {}

for cliente in clientes:
    caches[cliente] = {}

# Solicitudes
solicitudes = []
# Peticiones
peticiones = []

##########################################################
# Prefetching phase
##########################################################

# implementar prefetching de acuerdo con clique algorithm

K = len(clientes)
M = capacidad_cache

caches_np = matriz_np = np.zeros((M, K), dtype=int)

for cliente, cache in caches.items():
    # Seleccionar los archivos a almacenar en la caché
    # Implemento HPF
    archivos_cliente = sorted(archivos, key=lambda x: popularidad_por_archivo[x], reverse=True)[:capacidad_cache]
    cliente_int = int(cliente.split('_')[-1])
    #print(cliente_int)
    archivos_cliente_int = [int(archivo.split('_')[-1]) for archivo in archivos_cliente]
    caches_np[:, cliente_int-1] = archivos_cliente_int
    # Almacenar los archivos en la caché
    for archivo in archivos_cliente:
        caches[cliente][archivo] = {"timestamp": 0, "popularidad": popularidad_por_archivo[archivo]}
    
##########################################################
# Delivery phase
##########################################################

cache_hits = 0
cache_hits_list = []
num_solicitudes = 100
# Simulación del sistema
for i in range(num_solicitudes): # nº solicitudes
    # Resetear las variables
    peticiones = []
    requests = []
    cache_hits = 0
    # Generación de peticiones de archivo por cada cliente
    for cliente in clientes:
        archivo = random.choice(archivos) # Selecciono un archivo aleatorio
    # archivo = max(archivos, key=lambda x: popularidad_por_archivo[x]) # Selecciono el archivo más popular
        peticion = {"archivo": archivo, "cliente": cliente}
        
        # Compruebo si el cliente tiene cacheado dicho archivo
        if caches[cliente].get(archivo) is None:
            peticiones.append(peticion)
            
        else:
            # Hit en la caché
            #print(f"#{i+1}: Hit para {archivo} por {cliente}")
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
        requests.append(archivo if caches[cliente].get(archivo) is None else 0)
        requests_int = [int(x.split('_')[-1]) if isinstance(x, str) else x for x in requests]

    clique_indices = utils.find_largest_clique(requests_int, caches_np)
    
    # Compruebo si hay peticiones de archivos repetidos, para enviarlos en un solo mensaje
    archivos_a_enviar = utils.comprobar_peticiones_repetidas(peticiones)
    tiempo_inicio = i
    # Enviar mensajes a los clientes para entregar los archivos solicitados
    # Calcular el tiempo en enviar los archivos a los clientes
    tiempo_envio = utils.tiempo_envio_archivos(archivos_a_enviar, tasa_bits, size)
    
    # Actualizar las caches de los clientes:
    for peticion in peticiones:
        archivo = peticion["archivo"]
        cliente = peticion["cliente"]
        #print(f"#{i+1}: Fallo para {archivo} por {cliente}")

        # HPF: Actualización de la cache
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
    # pdb.set_trace()
  
#############################################################
# Plotear y printear resultados
#############################################################

print("Simulación finalizada.")

# Medición del rendimiento
tiempo_respuesta_promedio = utils.calcular_tiempo_respuesta_promedio(solicitudes)
tasa_aciertos_cache = utils.calcular_tasa_aciertos_cache(solicitudes)

print(f"Tiempo de respuesta promedio: {tiempo_respuesta_promedio}")
print(f"Tasa de aciertos en la caché: {tasa_aciertos_cache}")

# Plot the cache hits
plt.plot(range(num_solicitudes*capacidad_cache), cache_hits_list)
plt.xlabel('Número de solicitudes')
plt.ylabel('Aciertos de caché')
plt.title('Aciertos de caché vs. Número de solicitudes')
plt.show()


# Plotear los archivos en cada cache de cada cliente y su nivel de popularidad
for cliente, cache in caches.items():
    archivos = list(cache.keys())
    plt.figure()
    popularidades = [popularidad_por_archivo[archivo] for archivo in archivos]
    plt.bar(archivos, popularidades)
    plt.xlabel("Archivos")
    plt.ylabel("Popularidad")
    plt.title(f"Archivos en la caché de {cliente}")

#plt.show()