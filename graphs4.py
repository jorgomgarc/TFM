import random
import matplotlib.pyplot as plt
import utils
import numpy as np

def simulate(politica_prefetching, politica_delivery):
    ##########################################################
    # Parameters definition
    ##########################################################

    # Definición de archivos
    archivos = [f"file_{i}" for i in range(1, 31)]

    # Tamaño de los archivos
    size = 1000  # 10 bytes

    # Definición de clientes
    clientes = [f"cli_{i}" for i in range(1, 31)]

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
    # Peticiones
    peticiones = []

    ##########################################################
    # Placement phase
    ##########################################################

    K = len(clientes)
    M = capacidad_cache

    caches_np = np.zeros((M, K), dtype=int)

    for cliente, cache in caches.items():
        if politica_prefetching == "popularidad":
            archivos_cliente = sorted(archivos, key=lambda x: popularidad_por_archivo[x], reverse=True)[:capacidad_cache]
        elif politica_prefetching == "random":
            archivos_cliente = random.sample(archivos, min(len(archivos), capacidad_cache))  # Fill 'archivos_cliente' with random files

        cliente_int = int(cliente.split('_')[-1])
        archivos_cliente_int = [int(archivo.split('_')[-1]) for archivo in archivos_cliente]
        caches_np[:, cliente_int-1] = archivos_cliente_int
        # Almacenar los archivos en la caché
        for archivo in archivos_cliente:
            caches[cliente][archivo] = {"timestamp": 0, "popularidad": popularidad_por_archivo[archivo]}

    ##########################################################
    # Delivery phase
    ##########################################################

    cache_hits = 0
    satisfechos = 0
    cache_hits_list = []
    repetidos = 0
    num_solicitudes = 10000

    # Simulación del sistema
    for i in range(num_solicitudes): # nº solicitudes
        # Resetear las variables
        peticiones = []
        requests = []
        cache_hits = 0
        # Generación de peticiones de archivo por cada cliente 
        for cliente in clientes:
            if politica_delivery == "random":
                archivo = random.choice(archivos) # Selecciono un archivo aleatorio
            elif politica_delivery == "popularidad":
                archivo = random.choices(archivos, weights=[popularidad_por_archivo[x] for x in archivos])[0] # Selecciono un archivo con mayor probabilidad basado en su popularidad

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

            requests.append(archivo if caches[cliente].get(archivo) is None else 0)
            requests_int = [int(x.split('_')[-1]) if isinstance(x, str) else x for x in requests]

        
        clique_indices = utils.find_largest_clique(requests_int, caches_np)
        archivos_a_enviar = []
        archivos_solicitados = []
        _, repes = utils.comprobar_peticiones_repetidas(peticiones)
        repetidos += repes

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
        
        # Actualizar las caches de los clientes:
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

    #############################################################
    # Medición del rendimiento
    #############################################################

    tiempo_respuesta_promedio = utils.calcular_tiempo_respuesta_promedio(solicitudes)
    tasa_aciertos_cache = utils.calcular_tasa_aciertos_cache(solicitudes)
    ancho_de_banda = utils.calcular_ancho_de_banda(solicitudes, tasa_bits, size)

    return tiempo_respuesta_promedio, tasa_aciertos_cache, ancho_de_banda, satisfechos, repetidos

#############################################################
# Ejecución de la simulación para diferentes casos
#############################################################

# Definición de casos
cases = {
    "rand_rand": ("random", "random"),
    "rand_pop": ("random", "popularidad"),
    "pop_rand": ("popularidad", "random"),
    "pop_pop": ("popularidad", "popularidad")
}

# Resultados de la simulación
results = {}

for case, (prefetch, delivery) in cases.items():
    print(f"Running simulation for {case}...")
    avg_response_time, cache_hit_rate, bandwidth, satisfechos, repetidos = simulate(prefetch, delivery)
    results[case] = {
        "avg_response_time": avg_response_time,
        "cache_hit_rate": cache_hit_rate,
        "bandwidth": bandwidth,
        "satisfechos": satisfechos,
        "repetidos": repetidos
    }

# Print results
for case, result in results.items():
    print(f"\nResults for {case}:")
    print(f"Average response time: {result['avg_response_time']:.3f}")
    print(f"Cache hit rate: {result['cache_hit_rate']:.3f}")
    print(f"Bandwidth: {result['bandwidth']:.3f}")
    print(f"Satisfied requests: {result['satisfechos']}")
    print(f"Repeated requests: {result['repetidos']}")

# Plotting results
avg_response_times = [result["avg_response_time"] for result in results.values()]
cache_hit_rates = [result["cache_hit_rate"] for result in results.values()]
bandwidths = [result["bandwidth"] for result in results.values()]

cases_labels = list(results.keys())

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(cases_labels, avg_response_times, color='b')
plt.xlabel('Cases')
plt.ylabel('Average Response Time')
plt.title('Average Response Time for Different Cases')

plt.subplot(1, 2, 2)
plt.bar(cases_labels, bandwidths, color='r')
plt.xlabel('Cases')
plt.ylabel('Consumed Bandwidth')
plt.title('Consumed Bandwidth for Different Cases')

plt.tight_layout()
plt.show()
