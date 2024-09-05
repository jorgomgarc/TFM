########################################################################################################
# Plotear para uncoded uncoded. Con abos algoritmos que funcionan por popularidad
########################################################################################################


import random
import matplotlib.pyplot as plt
import utils

##########################################################
# Parameters definition
##########################################################

# Definición de archivos
archivos = [f"file_{i}" for i in range(1, 101)]

# Tamaño de los archivos
size = 1000  # 10 bytes

# Definición de clientes
clientes = [f"cli_{i}" for i in range(1, 31)]



# Definición del canal
tasa_bits = 1000000 # 1 Mbps

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


##########################################################
# Placement phase
##########################################################

prefetching_policy = "popularity"  # "random", "popularity"
num_solicitudes = 10000
politica_delivery = "popularidad"  # "random", "popularidad"

M_values = list(range(10, 51, 2))
results_cache_hits = []
results_cstisfied_requests = []

for M in M_values:
    solicitudes = []
    # Peticiones
    peticiones = []

    for cliente, cache in caches.items():
        if prefetching_policy == "popularity":
            archivos_cliente = sorted(archivos, key=lambda x: popularidad_por_archivo[x], reverse=True)[:M]
        elif prefetching_policy == "random":
            archivos_cliente = random.sample(archivos, min(len(archivos), M))  # Fill cache with random files

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
    for i in range(num_solicitudes): # nº solicitudes
        # Resetear las variables
        peticiones = []
        cache_hits = 0
        # Generación de peticiones de archivo por cada cliente
        for cliente in clientes:
            if politica_delivery == "random":
                archivo = random.choice(archivos) # Selecciono un archivo aleatorio
            elif politica_delivery == "popularidad":
                archivo = random.choices(archivos, weights=[popularidad_por_archivo[x] for x in archivos])[0] # Selecciono un archivo con mayor probabilidad basado en su popularidad
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
            
            # HPF
            # Si la caché está llena, eliminar el archivo menos popular
            if len(caches[cliente]) >= M:
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
    tasa_aciertos_cache, n = utils.calcular_tasa_aciertos_cache(solicitudes)
    # Medición del rendimiento
    print(f"Transmitted Packets: {satisfechos}")
    results_cache_hits.append(n)
    print(f"Repeated Requests: {repetidos}")
    results_cstisfied_requests.append(satisfechos)

# import pdb; pdb.set_trace()
 # Plotting the results
plt.figure(figsize=(12, 8))
plt.plot(M_values, results_cache_hits, label='Cache Hits')
plt.plot(M_values, results_cstisfied_requests, label='Transmitted Packets')

plt.xlabel('M (Files per cache)')
plt.ylabel('Transmitted Packets')
plt.title('Comparison between transmitted packets and cache hits')
plt.legend()
plt.grid(True)
plt.show()