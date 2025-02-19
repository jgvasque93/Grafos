import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import random
from matplotlib import cm
import numpy as np
from adjustText import adjust_text  # Importar la librería
import pandas as  pd
# # Crear un grafo con estructura de comunidad
# G = nx.barabasi_albert_graph(400, 3, seed=42)

# # Generar atributos aleatorios para los nodos
# generos = ["Masculino", "Femenino", "No Binario"]
# generos_musicales = ["Rock", "Pop", "Jazz", "Clásica", "Electrónica"]

# for node in G.nodes:
#     G.nodes[node]["genero"] = random.choice(generos)
#     G.nodes[node]["edad"] = random.randint(18, 60)
#     G.nodes[node]["genero_musical"] = random.choice(generos_musicales)

# Simulación de datos
num_users = 3000
num_topics = 30

# Generar usuarios y tópicos
usernames = [f'user{random.randint(0, 300)}' for i in range(1, num_users + 1)]
topicos = [f'Topico_{i}' for i in range(1, num_topics + 1)]
username_influencers = [random.choice(usernames) for _ in range(num_users)]
generos = ["Masculino", "Femenino", "No Binario"]
generos_musicales = ["Rock", "Pop", "Jazz", "Clásica", "Electrónica"]

# Crear DataFrame con la estructura correcta
data = pd.DataFrame({
    'username': usernames,
    'topico_conectar': [random.choice(topicos) for _ in range(num_users)],
    'username_influencer': username_influencers,
    'genero': [random.choice(generos) for _ in range(num_users)],
    'edad': [random.randint(18, 60) for _ in range(num_users)],
    'genero_musical': [random.choice(generos_musicales) for _ in range(num_users)]
})
# Crear el grafo
G = nx.Graph()

# Agregar nodos de usuarios
data.apply(lambda row: G.add_node(row['username'], tipo='user', **row.to_dict()), axis=1)

# Agregar nodos de tópicos
data['topico_conectar'].unique()
for topico in topicos:
    G.add_node(topico, tipo='topico')

# Agregar aristas (conexiones entre usuarios y tópicos)
data.apply(lambda row: G.add_edge(row['username'], row['topico_conectar']), axis=1)


#--------------
# Estado persistente en Streamlit
if "partition" not in st.session_state:
    st.session_state.partition = community_louvain.best_partition(G,resolution=1.5)

partition = st.session_state.partition


if "pos" not in st.session_state:
    unique_clusters = sorted(set(partition.values()))
    cluster_graphs = {c: nx.Graph() for c in unique_clusters}

    # Separar el grafo en subgrafos por clúster
    for node, cluster in partition.items():
        cluster_graphs[cluster].add_node(node)

    for u, v in G.edges():
        if partition[u] == partition[v]:  # Solo agregar conexiones internas
            cluster_graphs[partition[u]].add_edge(u, v)

    # Generar posiciones centradas para cada clúster
    cluster_positions = {}
    for cluster in unique_clusters:
        cluster_positions[cluster] = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) * 2  # Más centrado

    pos = {}
    for cluster, subgraph in cluster_graphs.items():
        cluster_pos = nx.spring_layout(subgraph, k=0.5, seed=42)  # Ajustamos la fuerza de atracción
        for node, coords in cluster_pos.items():
            pos[node] = cluster_positions[cluster] + coords * 0.7  # Reducimos la dispersión

    st.session_state.pos = pos

pos = st.session_state.pos

# Lista de clústeres ordenados
unique_clusters = sorted(set(partition.values()))
clusters = ["Todos"] + [f"Clúster {c}" for c in unique_clusters]

# Opciones de filtros
generos_opciones = ["Todos"] + generos
generos_musicales_opciones = ["Todos"] + generos_musicales

# Sidebar con opciones de filtro
st.sidebar.header("Filtros")

selected_cluster = st.sidebar.selectbox("Selecciona un clúster", clusters)
selected_genero = st.sidebar.selectbox("Selecciona un género", generos_opciones)
selected_genero_musical = st.sidebar.selectbox("Selecciona un género musical", generos_musicales_opciones)

# Determinar nodos a mostrar
filtered_nodes = set(G.nodes)

if selected_cluster != "Todos":
    cluster_id = unique_clusters[clusters.index(selected_cluster) - 1]
    filtered_nodes = {node for node in filtered_nodes if partition[node] == cluster_id}

if selected_genero != "Todos":
    filtered_nodes = {node for node in filtered_nodes if G.nodes[node]['tipo']=='user' and G.nodes[node]["genero"] == selected_genero}

if selected_genero_musical != "Todos":
    filtered_nodes = {node for node in filtered_nodes if G.nodes[node]['tipo']=='user' and G.nodes[node]["genero_musical"] == selected_genero_musical}

# Crear mapa de colores para nodos con una paleta personalizada
colors = cm.tab20.colors  
color_map = {c: colors[i % len(colors)] for i, c in enumerate(unique_clusters)}

# Tamaño de nodos basado en el grado de conexión
node_sizes = {n: G.degree(n) * random.randint(5, 60) for n in G.nodes}

# Identificar los nodos más grandes (top 10)
top_nodes = sorted(node_sizes, key=node_sizes.get, reverse=True)[:20]  

# Asignar colores a los nodos
node_colors = [color_map[partition[node]] if node in filtered_nodes else "lightgray" for node in G.nodes]

# Dibujar el grafo
fig, ax = plt.subplots(figsize=(11, 9))

nx.draw(
    G, pos, with_labels=False, node_color=node_colors,  width=0.05,
    alpha=0.5, node_size=[node_sizes[n] for n in G.nodes]
)
edge_colors = [
    color_map[partition[u]] if partition[u] == partition[v] else color_map[partition[u]]  # Asignar color del clúster de la arista
    for u, v in G.edges
]
# Dibujar las aristas con colores basados en los clústeres
nx.draw_networkx_edges(
    G, pos, edgelist=G.edges, edge_color=edge_colors, width=0.2, alpha=0.7
)

# Dibujar nodos con contorno negro
nx.draw_networkx_nodes(
    G, pos, nodelist=filtered_nodes, node_color=[color_map[partition[n]] for n in filtered_nodes],
    node_size=[node_sizes[n] for n in filtered_nodes], alpha=1, edgecolors="white", linewidths=0.5
)

# # Dibujar etiquetas en los nodos más grandes
# labels = {node: 'Interes'+str(node) for node in top_nodes}
label_sizes = {node: min(14, max(6, node_sizes[node]/80 )) for node in top_nodes}  # Tamaño de letra ajustado
# for node, text in labels.items():
#     ax.text(pos[node][0], pos[node][1], text, fontsize=label_sizes[node], 
#             fontweight="bold", ha="center", va="center", color="black")

# Dibujar etiquetas en los nodos más grandes
texts = []
for node in top_nodes:
    text = ax.text(pos[node][0], pos[node][1], f"Interes {node}",
                   fontsize=label_sizes[node], fontweight="bold",
                   ha="center", va="center", color="black")
    texts.append(text)

# Ajustar las etiquetas para evitar superposición
adjust_text(texts, expand_points=(2, 2), force_text=(0.3, 0.3), arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

# Mostrar información de los nodos filtrados
st.sidebar.subheader("Nodos filtrados:")
for node in filtered_nodes:
    info = G.nodes[node]
    if(info['tipo']=='user'):
        st.sidebar.text(f"Nodo {node}: {info['genero']}, {info['edad']} años, {info['genero_musical']}")

# Mostrar gráfico
st.pyplot(fig)
