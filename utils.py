import collections
import random
import numpy as np
import pdb

def generar_popularidad_random():
    # Se podría cambiar y utilizar otra distribución: la normal o la que sea
    # Uniform popularity distribution
    return random.randint(1, 10)

def rellenar_archivo(archivo, size):
    letras = list("abcdefghijklmnñopqrstuvwxyz")
    random.shuffle(letras)
    with open("files/"+archivo, "wb") as f:
        for _ in range(size):
            if letras:  # Check if letras is not empty before popping
                letra = random.choice(letras).encode("utf-8")
                f.write(letra)

def descargar_archivo_del_almacenamiento_principal(size, tasa_bits):

    # Calcular el tiempo de descarga
    tiempo_descarga = size / tasa_bits

    # Agregar latencia y variabilidad
    tiempo_descarga += random.uniform(0.1, 0.5)  # Segundos

    return tiempo_descarga

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

    return tasa_aciertos_cache

def comprobar_peticiones_repetidas(peticiones):
    # Obtener la lista de archivos y clientes de las peticiones
    archivos = [peticion["archivo"] for peticion in peticiones]

    # Comprobar si hay archivos o clientes repetidos
    archivos_repetidos = [item for item, count in collections.Counter(archivos).items() if count > 1]

    # Eliminar los archivos y clientes repetidos
    archivos_sin_repetir = list(set(archivos))

    return archivos_sin_repetir

def tiempo_envio_archivos(archivos_a_enviar, tasa_bits, size):
    # Simulación del envío del archivo al cliente
    total_size = sum([size for archivo in archivos_a_enviar])
    tiempo_envio = total_size / tasa_bits  # Segundos
    return tiempo_envio


################################################
# Uncoded - Coded
################################################

def dividir_archivo(archivo, numero_partes):
    partes = []
    for i in range(numero_partes):
        contenido = archivo
        parte = contenido[i::numero_partes]
        partes.append(parte)
    return partes

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



# This function takes a ROW K-VECTOR of requests from the different users
# (0 for no request) and a MxK matrix of cache contents and finds a
# largest clique, returning the indices of the corresponding users
def find_largest_clique(requests, caches):
    # Discard users without any request
    aux = [i for i, req in enumerate(requests) if req != 0]
    requests_aux = [0 for _ in range(len(aux))]
    caches_aux = [[0 for _ in range(len(aux))] for _ in range(len(caches))]
    for i in range(len(aux)):
        requests_aux[i] = requests[aux[i]]
        
    caches_aux = [[fila[i] for i in aux] for fila in caches]


    usrs_with_req = aux
    n_left = len(aux)  # number of users remaining
    max_clique_len = 0
    usr_clique = []

    for i in range(n_left):  # loop over users i
        req_i = requests_aux[i]  # request from current user
        usr_candidates = []  # This will store the candidates to form a clique with current user
        caches_aux = np.array(caches_aux)

        for j in range(i + 1, caches_aux.shape[1]):  # Loop over users later than i
            if any(caches_aux[:, j] == req_i) or requests_aux[j] == req_i:  # Do nothing unless user j stores or demands req_i
                req_j = requests_aux[j]
                if any(caches_aux[:, i] == req_j) or req_i == req_j:  # If current user stores or demands req_j...
                    usr_candidates.append(j)  # ...store as viable candidate
                    
        # Find largest clique that includes current user i
        if not usr_candidates:
            clique_i = [usrs_with_req[i]]
        elif max_clique_len >= 1 + len(usr_candidates):
            clique_i = []  # I already have a clique larger than all the candidates
        else:
            #pdb.set_trace()
            selected_requests = [requests_aux[i] for i in usr_candidates]
            aux = find_largest_clique(selected_requests, caches_aux[:, usr_candidates])
            aux = [usr_candidates[k] for k in aux]
            clique_i = [usrs_with_req[i]] + aux

        # If current clique is largest so far, store it in usr_clique (output)
        if max_clique_len < len(clique_i):
            usr_clique = clique_i
            max_clique_len = len(clique_i)

    return usr_clique
