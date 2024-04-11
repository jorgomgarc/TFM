import random
import numpy as np

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

def descargar_archivo_del_almacenamiento_principal(size, tasa_bits):

    # Calcular el tiempo de descarga
    tiempo_descarga = size / tasa_bits

    # Agregar latencia y variabilidad
    tiempo_descarga += random.uniform(0.1, 0.5)  # Segundos

    return tiempo_descarga

def tiempo_envio_archivo(size, tasa_bits):
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

    # Plotear la tasa de aciertos en la caché
    # plt.figure()
    # plt.plot(numero_aciertos_cache)
    # plt.xlabel("Solicitudes")
    # plt.ylabel("Tasa de aciertos en la caché")
    # plt.title("Tasa de aciertos en la caché")
    # plt.show()

    return tasa_aciertos_cache




# This function takes a ROW K-VECTOR of requests from the different users
# (0 for no request) and a MxK matrix of cache contents and finds a
# largest clique, returning the indices of the corresponding users
def find_largest_clique(requests, caches):
    # Discard users without any request
    aux = [i for i, req in enumerate(requests) if req != 0]
    requests = requests[aux]
    caches = caches[:, aux]
    usrs_with_req = aux
    n_left = len(aux)  # number of users remaining

    max_clique_len = 0
    usr_clique = []

    for i in range(n_left):  # loop over users i
        req_i = requests[i]  # request from current user
        usr_candidates = []  # This will store the candidates to form a clique with current user

        for j in range(i + 1, caches.shape[1]):  # Loop over users later than i
            if any(caches[:, j] == req_i) or requests[j] == req_i:  # Do nothing unless user j stores or demands req_i
                req_j = requests[j]
                if any(caches[:, i] == req_j) or req_i == req_j:  # If current user stores or demands req_j...
                    usr_candidates.append(j)  # ...store as viable candidate

        # Find largest clique that includes current user i
        if not usr_candidates:
            clique_i = [usrs_with_req[i]]
        elif max_clique_len >= 1 + len(usr_candidates):
            clique_i = []  # I already have a clique larger than all the candidates
        else:
            aux = find_largest_clique(requests[usr_candidates], caches[:, usr_candidates])
            aux = [usr_candidates[k] for k in aux]
            clique_i = [usrs_with_req[i]] + aux

        # If current clique is largest so far, store it in usr_clique (output)
        if max_clique_len < len(clique_i):
            usr_clique = clique_i
            max_clique_len = len(clique_i)

    return usr_clique
