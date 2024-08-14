import random
import matplotlib.pyplot as plt
import utils

def simular(prefetching_policy, politica_delivery, num_solicitudes=10000):
    # Definición de archivos
    archivos = [f"file_{i}" for i in range(1, 21)]
    size = 1000  # 10 bytes
    clientes = [f"cli_{i}" for i in range(1, 31)]
    capacidad_cache = 10
    tasa_bits = 1000000  # 1 Mbps

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

    caches = {cliente: {} for cliente in clientes}

    solicitudes = []

    for cliente, cache in caches.items():
        if prefetching_policy == "popularity":
            archivos_cliente = sorted(archivos, key=lambda x: popularidad_por_archivo[x], reverse=True)[:capacidad_cache]
        elif prefetching_policy == "random":
            archivos_cliente = random.sample(archivos, min(len(archivos), capacidad_cache))  # Fill cache with random files

        for archivo in archivos_cliente:
            caches[cliente][archivo] = {"timestamp": 0, "popularidad": popularidad_por_archivo[archivo]}

    cache_hits = 0
    satisfechos = 0
    repetidos = 0
    cache_hits_list = []
    
    for i in range(num_solicitudes):
        peticiones = []
        cache_hits = 0
        
        for cliente in clientes:
            if politica_delivery == "random":
                archivo = random.choice(archivos) # Selecciono un archivo aleatorio
            elif politica_delivery == "popularidad":
                archivo = random.choices(archivos, weights=[popularidad_por_archivo[x] for x in archivos])[0] # Selecciono un archivo con mayor probabilidad basado en su popularidad
            
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
            
        archivos_a_enviar, repes = utils.comprobar_peticiones_repetidas(peticiones)
        satisfechos += len(archivos_a_enviar)
        repetidos += repes
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

    tiempo_respuesta_promedio = utils.calcular_tiempo_respuesta_promedio(solicitudes)
    ancho_de_banda_consumido = sum([solicitud['bytes_transferidos'] for solicitud in solicitudes]) / num_solicitudes

    return tiempo_respuesta_promedio, ancho_de_banda_consumido

# Simulaciones para cada combinación
resultados = {}
configuraciones = [
    ("popularity", "random"),
    ("popularity", "popularidad"),
    ("random", "random"),
    ("random", "popularidad")
]

for prefetching_policy, politica_delivery in configuraciones:
    tiempo_respuesta_promedio, ancho_de_banda_consumido = simular(prefetching_policy, politica_delivery)
    resultados[(prefetching_policy, politica_delivery)] = {
        "tiempo_respuesta_promedio": tiempo_respuesta_promedio,
        "ancho_de_banda_consumido": ancho_de_banda_consumido
    }

# Plotting
labels = ["Pop-Pop", "Pop-Rand", "Rand-Pop", "Rand-Rand"]
response_times = [resultados[(p, d)]["tiempo_respuesta_promedio"] for p, d in configuraciones]
bandwidths = [resultados[(p, d)]["ancho_de_banda_consumido"] for p, d in configuraciones]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(labels, response_times)
plt.xlabel("Configurations")
plt.ylabel("Average Response Time")
plt.title("Average Response Time for Different Configurations")

plt.subplot(1, 2, 2)
plt.bar(labels, bandwidths)
plt.xlabel("Configurations")
plt.ylabel("Consumed Bandwidth")
plt.title("Consumed Bandwidth for Different Configurations")

plt.tight_layout()
plt.show()
