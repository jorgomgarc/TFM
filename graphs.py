import random
import matplotlib.pyplot as plt
import utils
import numpy as np

def ejecutar_simulacion(capacidad_cache):
    ##########################################################
    # Parameters definition
    ##########################################################

    # Definición de archivos
    archivos = [f"file_{i}" for i in range(1, 101)]

    # Tamaño de los archivos
    size = 1000  # 10 bytes
    # Rellenar los archivos
    for archivo in archivos:
        utils.rellenar_archivo(archivo, size)

    # Asignar popularidad a los archivos (de forma random)
    archivos_por_popularidad = {}
    popularidad_por_archivo = {}
    for archivo in archivos:
        popularidad = utils.generar_popularidad_random()
        popularidad_por_archivo[archivo] = popularidad
        if popularidad not in archivos_por_popularidad:
            archivos_por_popularidad[popularidad] = []
        archivos_por_popularidad[popularidad].append(archivo)

    # Definición de clientes
    clientes = [f"cli_{i}" for i in range(1, 31)]

    # Definición del servidor
    # capacidad_cache = 10  # Este valor se recibe como parámetro

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

    K = len(clientes)
    M = capacidad_cache

    caches_np = np.zeros((M, K), dtype=int)

    politica_prefetching = "popularidad"  # "random", "popularidad"

    for cliente, cache in caches.items():
        if politica_prefetching == "popularidad":
            archivos_cliente = sorted(archivos, key=lambda x: popularidad_por_archivo[x], reverse=True)[:capacidad_cache]
        elif politica_prefetching == "random":
            archivos_cliente = random.sample(archivos, min(len(archivos), capacidad_cache))
        cliente_int = int(cliente.split('_')[-1])
        archivos_cliente_int = [int(archivo.split('_')[-1]) for archivo in archivos_cliente]
        caches_np[:, cliente_int-1] = archivos_cliente_int
        for archivo in archivos_cliente:
            caches[cliente][archivo] = {"timestamp": 0, "popularidad": popularidad_por_archivo[archivo]}
    
    ##########################################################
    # Delivery phase
    ##########################################################

    cache_hits = 0
    satisfechos = 0
    cache_hits_list = []
    num_solicitudes = 10000
    politica_delivery = "popularidad"  # "random", "popularidad"

    for i in range(num_solicitudes): # nº solicitudes
        peticiones = []
        requests = []
        cache_hits = 0

        for cliente in clientes:
            if politica_delivery == "random":
                archivo = random.choice(archivos)
            elif politica_delivery == "popularidad":
                archivo = random.choices(archivos, weights=[popularidad_por_archivo[x] for x in archivos])[0]

            peticion = {"archivo": archivo, "cliente": cliente}
            if caches[cliente].get(archivo) is None:
                peticiones.append(peticion)
            else:
                caches[cliente][archivo]["timestamp"] = i
                tiempo_inicio = i  
                tiempo_fin = i
                hit_cache = True
                cache_hits += 1
                bytes_transferidos = 0
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
        archivos_a_enviar = []
        archivos_solicitados = []

        if len(clique_indices) > 1:
            if clique_indices[0] > clique_indices[1] or clique_indices[0] == clique_indices[1]:
                archivo_tonto = requests_int[clique_indices[0]]
                archivos_solicitados.append(f"file_{archivo_tonto}")
                archivos_solicitados = [archivos_solicitados[0]] + [peticiones[i]["archivo"] for i in clique_indices[1:]]
            else:
                archivos_solicitados = [peticiones[i]["archivo"] for i in clique_indices]

            archivos_repetidos = utils.comprobar_archivos_repetidos(archivos_solicitados)

            if archivos_repetidos:
                archivos_a_enviar, _ = utils.comprobar_peticiones_repetidas(peticiones)
            else:
                utils.xor_files(archivos_solicitados)
                archivo_xor = f"file_xor"
                archivos = [peticion["archivo"] for peticion in peticiones]
                archivos_a_enviar = [archivo for archivo in archivos if archivo not in archivos_solicitados] + [archivo_xor]
                archivos_a_enviar = list(set(archivos_a_enviar))

        satisfechos += len(archivos_a_enviar) 
        tiempo_inicio = i
        tiempo_envio = utils.tiempo_envio_archivos(archivos_a_enviar, tasa_bits, size)
        
        for peticion in peticiones:
            archivo = peticion["archivo"]
            cliente = peticion["cliente"]

            if len(caches[cliente]) >= capacidad_cache:
                archivo_menos_popular = min(caches[cliente].keys(), key=lambda x, cliente=cliente: caches[cliente][x]["popularidad"])
                del caches[cliente][archivo_menos_popular]
            caches[cliente][archivo] = {"timestamp": i, "popularidad": popularidad_por_archivo[archivo]}

            tiempo_descarga = utils.descargar_archivo_del_almacenamiento_principal(size, tasa_bits)

            tiempo_fin = i + tiempo_descarga + tiempo_envio
            hit_cache = False
            bytes_transferidos = size * len(archivos_a_enviar)

            solicitudes.append({
                "tiempo_inicio": tiempo_inicio,
                "tiempo_fin": tiempo_fin,
                "hit_cache": hit_cache,
                "bytes_transferidos": bytes_transferidos,
            })

    return satisfechos

capacidad_cache_values = range(10, 31)
satisfechos_values = []

for capacidad_cache in capacidad_cache_values:
    satisfechos = ejecutar_simulacion(capacidad_cache)
    satisfechos_values.append(satisfechos)

plt.plot(capacidad_cache_values, satisfechos_values)
plt.xlabel('Capacidad de Caché')
plt.ylabel('Número de Requests Satisfechas')
plt.title('Requests Satisfechas vs. Capacidad de Caché')
plt.show()
