import random
import matplotlib.pyplot as plt
import pdb
import utils

##########################################################
# Parameters definition
##########################################################

# Definición de archivos
archivos = [f"arch{i}" for i in range(1, 51)]

# Tamaño de los archivos
size = 1000  # 10 bytes

# Definición de clientes
clientes = [f"cli{i}" for i in range(1, 11)]

# Definición del servidor
capacidad_cache = 10

# Definición del canal
tasa_bits = 1000000 # 1 Mbps

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

# Plotear la distribución de popularidad por archivos
plt.figure()
for archivo, popularidad in popularidad_por_archivo.items():
    plt.bar(archivo, popularidad)
plt.xlabel("Archivos")
plt.ylabel("Popularidad")
plt.title("Popularidad de archivos")
plt.show(block=False)

# Política de prefetching
politica_prefetching = "LFU"  # "LRU", "LFU", "HPF
frecuencia = {}

# Diccionario con caches de los clientes
caches = {}

for cliente in clientes:
    caches[cliente] = {}
    frecuencia[cliente] = {}

# Solicitudes
solicitudes = []
# Peticiones
peticiones = []

# Estado de las caches (lo que conoce el servidor):
servidor = {}
for cliente in clientes:
    servidor[cliente] = {}
    for archivo in archivos:
            servidor[cliente][archivo] = 0

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
        servidor[cliente][archivo] = 1

# Plotear los archivos en cada cache de cada cliente ANTES de delivery phase
# for cliente, cache in caches.items():
#     archivos_cache_cliente = list(cache.keys())
#     plt.figure()
#     popularidades = [popularidad_por_archivo[archivo] for archivo in archivos_cache_cliente]
#     plt.bar(archivos_cache_cliente, popularidades)
#     plt.xlabel("Archivos")
#     plt.ylabel("Popularidad")
#     plt.title(f"Archivos en la caché de {cliente}")
#     plt.show(block=False)

##########################################################
# Delivery phase
##########################################################

# Simulación del sistema
for i in range(10): # nº solicitudes
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
            print(f"#{i+1}: Hit para {archivo} por {cliente}")
            caches[cliente][archivo]["timestamp"] = i
            # Actualizo la frecuencia del archivo
            # Si el archivo ya está en el diccionario, incrementa su frecuencia
            if archivo in frecuencia[cliente]:
                frecuencia[cliente][archivo] += 1
            # Si el archivo no está en el diccionario, añádelo con frecuencia 1
            else:
                frecuencia[cliente][archivo] = 1
            
            # Fin de la solicitud
            tiempo_fin = i
            hit_cache = True
            bytes_transferidos = size

    print(peticiones)
    pdb.set_trace()
    # Compruebo si hay peticiones de archivos repetidos, para enviarlos en un solo mensaje
    peticiones = utils.comprobar_peticiones_repetidas(peticiones)
    print(peticiones)
    pdb.set_trace()
    # Enviar mensajes a los clientes para entregar los archivos solicitados
    for peticion in peticiones:
        archivo = peticion["archivo"]
        cliente = peticion["cliente"]
        if servidor[cliente][archivo] == 1:
            print(f"Enviando archivo {archivo} a {cliente}")
            # Aquí iría el código para enviar el archivo al cliente
        else:
            print(f"El archivo {archivo} no está disponible en el servidor para {cliente}")
    # Inicio de la solicitud
    tiempo_inicio = i
    
    
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

        # Actualización de la cache
        if politica_prefetching == "LFU":
            # Se elimina el archivo con menor frecuencia
            if len(caches[cliente]) >= capacidad_cache:
                archivo_lfu = min(caches[cliente].keys(), key=lambda x, cliente=cliente: caches[cliente][x].get("frecuencia", 0))
                del caches[cliente][archivo_lfu]

            caches[cliente][archivo] = {"frecuencia": frecuencia[cliente][archivo]}

        elif politica_prefetching == "HPF":
            # Prefetch del archivo más popular
            
            # Si la caché está llena, eliminar el archivo menos popular
            if len(caches[cliente]) >= capacidad_cache:
                archivo_menos_popular = min(caches[cliente].keys(), key=lambda x, cliente=cliente: caches[cliente][x]["popularidad"])
                del caches[cliente][archivo_menos_popular]

            # Agregar el archivo a la caché y establecer su popularidad
            caches[cliente][archivo] = {"timestamp": i, "popularidad": popularidad_por_archivo[archivo]}


        # Descarga del archivo del almacenamiento principal
        tiempo_descarga = utils.descargar_archivo_del_almacenamiento_principal(size, tasa_bits)

        # Envío del archivo al cliente
        tiempo_envio = utils.tiempo_envio_archivo(size, tasa_bits)

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
  
#############################################################
# Plotear y printear resultados
#############################################################

print("Simulación finalizada.")

# Medición del rendimiento
tiempo_respuesta_promedio = utils.calcular_tiempo_respuesta_promedio(solicitudes)
tasa_aciertos_cache = utils.calcular_tasa_aciertos_cache(solicitudes)

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