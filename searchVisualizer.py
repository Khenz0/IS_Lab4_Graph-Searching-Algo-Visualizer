import queue
import random
import networkx as nx
import matplotlib.pyplot as plt
import time
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox, simpledialog

def dfs(graph, start_node, end_node, visited=None, path=None, cost=0):
    # Initialize visited set and path list if not provided
    if visited is None:
        visited = set()
    if path is None:
        path = []

    # Initialize the order list to keep track of the exploration order
    order = []

    # Check if the start node has not been visited
    if start_node not in visited:
        # Append the current state (node, path, cost) to the order list
        order.append((start_node, path.copy(), cost))
        visited.add(start_node)
        # Stop the search if the end node is found
        if start_node == end_node:
            return order

        # Explore neighbors of the current node
        for node in graph[start_node]:
            if node not in visited:
                # Recursive call to explore the neighbor
                order.extend(dfs(graph, node, end_node, visited, path + [start_node], cost + graph[start_node][node]['weight']))
                # Stop the search if the end node is found
                if order[-1][0] == end_node:
                    break

    return order

def bfs(graph, start_node, end_node):
    # Initialize visited set and a queue with the starting node, path, and cost
    visited = set()
    q = queue.Queue()
    q.put((start_node, [start_node], 0))  # Enqueue tuple (node, path, cost)
    order = []

    # Continue the search while the queue is not empty
    while not q.empty():
        node, path, cost = q.get()
        if node not in visited:
            # Append the current state (node, path, cost) to the order list
            order.append((node, path.copy(), cost))
            visited.add(node)
            # Stop the search if the end node is found
            if node == end_node:
                break

            # Enqueue neighbors of the current node
            for neighbor in graph[node]:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_cost = cost + graph[node][neighbor]['weight']
                    q.put((neighbor, new_path, new_cost))
    return order

def hill_climbing(graph, start_node, end_node, heuristic):
    # Initialize current node, path, and cost
    current_node = start_node
    path = [current_node]
    cost = 0

    # Initialize the order list to keep track of the exploration order
    order = [(current_node, path.copy(), cost)]

    # Continue the search until the current node is the end node
    while current_node != end_node:
        # Get neighbors and calculate heuristic values
        neighbors = list(graph[current_node])
        neighbors.sort(key=lambda neighbor: heuristic(neighbor, end_node))

        found = False
        # Iterate through neighbors to find the one with the minimum heuristic value
        for neighbor in neighbors:
            if neighbor not in path:
                # Update path and cost based on the selected neighbor
                path.append(neighbor)
                cost += heuristic(neighbor, end_node)
                current_node = neighbor
                found = True
                # Append the current state (node, path, cost) to the order list
                order.append((current_node, path.copy(), cost))
                break

        if not found:
            break  # Stuck in a local maximum, exit the loop

    return order

def beam_search(graph, start_node, end_node, beam_width, heuristic):
    # Set to keep track of visited nodes
    visited = set()

    # Queue for BFS traversal
    q = queue.Queue()

    # Initialize the queue with the start node, path, and cost
    q.put((start_node, [start_node], 0))

    # List to store the order of nodes visited
    order = []

    # Continue traversal until the queue is empty
    while not q.empty():
        # List to store nodes at the current level
        current_level = []

        # Process nodes at the current level, up to the beam width
        for _ in range(min(beam_width, q.qsize())):
            # Get the node, path, and cost from the front of the queue
            node, path, cost = q.get()

            # Check if the node has not been visited
            if node not in visited:
                # Add the node, path, and cost to the order list
                order.append((node, path.copy(), cost))
                # Mark the node as visited
                visited.add(node)

                # Check if the current node is the goal node
                if node == end_node:
                    break

                # Get neighbors of the current node and sort them based on the heuristic
                neighbors = list(graph[node])
                neighbors.sort(key=lambda neighbor: heuristic(neighbor, end_node))

                # Explore neighbors and add them to the current level
                for neighbor in neighbors:
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        new_cost = cost + heuristic(neighbor, end_node)
                        current_level.append((neighbor, new_path, new_cost))

        # Add nodes at the current level to the queue for further exploration
        for entry in current_level:
            q.put(entry)

    # Return the order in which nodes were visited
    return order

def branch_and_bound(graph, start_node, end_node):
    # Initialize priority queue with the starting node, path, and cost
    pq = queue.PriorityQueue()
    pq.put((0, (start_node, [start_node], 0)))  # Enqueue tuple (priority, (node, path, cost))
    visited = set()
    order = []

    # Continue the search while the priority queue is not empty
    while not pq.empty():
        # Retrieve the tuple from the priority queue, unpack its elements, and assign them to variables
        # The underscore (_) is used to indicate that we are intentionally ignoring the first element of the tuple
        _, (node, path, cost) = pq.get()
        if node not in visited:
            # Append the current state (node, path, cost) to the order list
            order.append((node, path.copy(), cost))
            visited.add(node)
            # Stop the search if the end node is found
            if node == end_node:
                break

            # Enqueue neighbors of the current node with priority based on cost (branching)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_cost = cost + graph[node][neighbor]['weight']
                    pq.put((new_cost, (neighbor, new_path, new_cost)))

    return order

def a_star(graph, start_node, end_node, heuristic):
    # Priority queue for A* search
    pq = queue.PriorityQueue()

    # Initialize with the starting node, path, and cost
    pq.put((0, (start_node, [start_node], 0)))  # Enqueue tuple (priority, (node, path, cost))
    visited = set()
    order = []

    while not pq.empty():
        _, (node, path, cost) = pq.get()

        # If the node has not been visited, process it
        if node not in visited:
            # Append the current state (node, path, cost) to the order list
            order.append((node, path.copy(), cost))
            visited.add(node)

            # If the goal node is reached, break out of the loop
            if node == end_node:
                break

            # Explore neighbors of the current node
            for neighbor in graph[node]:
                if neighbor not in visited:
                    # Update the path and cost based on the selected neighbor
                    new_path = path + [neighbor]
                    new_cost = cost + graph[node][neighbor]['weight']
                    
                    # Calculate the priority (total cost + heuristic) for the neighbor
                    priority = new_cost + heuristic(neighbor, end_node)
                    
                    # Enqueue the neighbor with its priority, path, and cost
                    pq.put((priority, (neighbor, new_path, new_cost)))

    return order

def generate_random_graph_dialog():
    try:
        num_vertices = simpledialog.askinteger("Input", "Enter the number of vertices:")
        num_edges = simpledialog.askinteger("Input", "Enter the number of edges:")
        
        if num_vertices is not None and num_edges is not None:
            G.clear()
            G.update(generate_random_graph(num_vertices, num_edges))
            update_graph()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def generate_random_graph(num_vertices, num_edges, weight_range=(1, 10)):
    # Check if the number of vertices is valid
    if num_vertices < 1:
        raise ValueError("Number of vertices must be greater than 0.")

    # Check if the number of edges is valid
    if num_edges < 0 or num_edges > num_vertices * (num_vertices - 1) // 2:
        raise ValueError("Number of edges is invalid.")

    # Create an empty graph
    G = nx.Graph()

    # Add vertices to the graph
    vertices = [chr(ord('A') + i) for i in range(num_vertices)]
    G.add_nodes_from(vertices)

    # Add random edges to the graph
    edges_added = 0
    while edges_added < num_edges:
        i, j = random.sample(range(num_vertices), 2)
        if not G.has_edge(vertices[i], vertices[j]):
            # Add an edge between vertex i and vertex j
            weight = random.randint(weight_range[0], weight_range[1])
            G.add_edge(vertices[i], vertices[j], weight=weight)
            edges_added += 1

    return G

def generate_random_connected_graph(num_vertices):
    # Generates a random connected graph using the Watts-Strogatz model.

    # Check if the number of vertices is valid
    if num_vertices < 1:
        raise ValueError("Number of vertices must be greater than 0.")

    # Parameters for the Watts-Strogatz model
    k = 4  # Average degree (adjust as needed)
    p = 0.1  # Probability of rewiring (adjust as needed)

    # Generate a connected graph using the Watts-Strogatz model
    G = nx.connected_watts_strogatz_graph(num_vertices, k, p)

    # Relabel nodes to use single-letter labels (A, B, C, ...)
    mapping = {i: chr(ord('A') + i) for i in range(num_vertices)}
    G = nx.relabel_nodes(G, mapping)

    # Assign random weights to edges
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(1, 10)  # You can adjust the weight range

    return G

# Global variable to control visualization pause/play
visualization_paused = False

# Function to visualize the search process without heuristic
def visualization(order, title, G, pos, animation_speed):
    global visualization_paused
    plt.figure()

    visited_nodes = []  # List to keep track of visited nodes
    enqueues = 0       # Counter for the number of enqueues

    # Iterate through each step in the search order
    for step in order:
        node, path, cost = step
        visited_nodes.append(node)  # Add the current node to the visited nodes list

        # Create a string of visited nodes, node colors, and edge labels for visualization
        visited_sequence = ', '.join(visited_nodes)
        node_colors = ['yellow' if x == node else 'skyblue' for x in G.nodes()]
        edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}

        # Clear the previous plot, set title, and draw the graph with updated information
        plt.clf()
        plt.title(title, loc="left", fontsize=10, pad=5)
        nx.draw(G, pos, with_labels=True, node_color=node_colors)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')

        # Display information on the plot
        info_text = f"Number of Enqueues: {enqueues}\nPath: {' -> '.join(path + [node])}\nQueue Size: {len(path)}\nPath Elements: {visited_sequence}\nTotal Path Cost: {cost}"
        plt.text(0, 1.0, info_text, verticalalignment='top', horizontalalignment='left', transform=plt.gca().transAxes, fontsize=9)

        plt.draw()
        plt.pause(animation_speed)  # Pause for animation speed

        # Check if visualization is paused and wait until resumed
        if visualization_paused:
            while visualization_paused:
                plt.pause(0.1)

        # Increment the enqueues counter when a new node is enqueued
        if len(path) > 1 and path[-1] != path[-2]:
            enqueues += 1

    plt.show()
    time.sleep(0.5)

# Function to visualize the search process with heuristic
def visualization_with_heuristic(order, title, G, pos, heuristic, end_node, animation_speed):
    global visualization_paused
    plt.figure()

    visited_nodes = []  # List to keep track of visited nodes
    enqueues = 0       # Counter for the number of enqueues

    # Iterate through each step in the search order
    for step in order:
        node, path, cost = step
        visited_nodes.append(node)  # Add the current node to the visited nodes list

        # Create a string of visited nodes, node colors, and edge labels for visualization
        visited_sequence = ', '.join(visited_nodes)
        node_colors = ['yellow' if x == node else 'skyblue' for x in G.nodes()]
        edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}

        # Clear the previous plot, set title, and draw the graph with updated information
        plt.clf()
        plt.title(title, loc="left", fontsize=10, pad=5)
        nx.draw(G, pos, with_labels=True, node_color=node_colors)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')

        # Display information on the plot, including heuristic value
        heuristic_value = heuristic(node, end_node) if hasattr(heuristic, '__call__') else 0
        info_text = f"Number of Enqueues: {enqueues}\nPath: {' -> '.join(path + [node])}\nQueue Size: {len(path)}\nPath Elements: {visited_sequence}\nTotal Path Cost: {cost}\nHeuristic: {heuristic_value:.2f}"
        plt.text(0, 1.0, info_text, verticalalignment='top', horizontalalignment='left', transform=plt.gca().transAxes, fontsize=9)

        plt.draw()
        plt.pause(animation_speed)  # Pause for animation speed

        # Check if visualization is paused and wait until resumed
        if visualization_paused:
            while visualization_paused:
                plt.pause(0.1)

        # Increment the enqueues counter when a new node is enqueued
        if len(path) > 1 and path[-1] != path[-2]:
            enqueues += 1

    plt.show()
    time.sleep(0.5)


def visualize_dfs_dialog():
    start_node = simpledialog.askstring("Input", "Enter the start node:")
    end_node = simpledialog.askstring("Input", "Enter the end node:")
    speed = max(0.1, min(1.0, 1.0 - (int(speed_var.get()) - 1) / 9))
    if start_node and end_node:
        start_node = start_node.upper()
        end_node = end_node.upper()
        visualization(dfs(G, start_node, end_node), 'DFS Visualization', G, pos, speed)

def visualize_bfs_dialog():
    start_node = simpledialog.askstring("Input", "Enter the start node:")
    end_node = simpledialog.askstring("Input", "Enter the end node:")
    speed = max(0.1, min(1.0, 1.0 - (int(speed_var.get()) - 1) / 9))
    if start_node and end_node:
        start_node = start_node.upper()
        end_node = end_node.upper()
        visualization(bfs(G, start_node, end_node), 'BFS Visualization', G, pos, speed)

# Define your heuristic function
def my_heuristic(node, goal_node):
    # Manhattan distance heuristic
    return abs(ord(node) - ord(goal_node))

def visualize_hill_climbing_dialog():
    start_node = simpledialog.askstring("Input", "Enter the start node:")
    end_node = simpledialog.askstring("Input", "Enter the end node:")
    speed = max(0.1, min(1.0, 1.0 - (int(speed_var.get()) - 1) / 9))
    
    if start_node and end_node:
        start_node = start_node.upper()
        end_node = end_node.upper()

        # Call the hill climbing with heuristic function
        order = hill_climbing(G, start_node, end_node, my_heuristic)

        visualization_with_heuristic(order, 'Hill Climbing Visualization', G, pos, my_heuristic, end_node, speed)

def visualize_beam_search_dialog():
    start_node = simpledialog.askstring("Input", "Enter the start node:")
    end_node = simpledialog.askstring("Input", "Enter the end node:")
    beam_width = simpledialog.askinteger("Input", "Enter the beam width:")
    speed = max(0.1, min(1.0, 1.0 - (int(speed_var.get()) - 1) / 9))

    if start_node and end_node and beam_width is not None:
        start_node = start_node.upper()
        end_node = end_node.upper()

        # Call the beam search with heuristic function
        order = beam_search(G, start_node, end_node, beam_width, my_heuristic)

        # Display the beam width in the title
        title = f'Beam Search Visualization (Beam Width: {beam_width})'
        
        # Pass end_node to the visualization_with_heuristic function
        visualization_with_heuristic(order, title, G, pos, my_heuristic, end_node, speed)

def visualize_branch_and_bound_dialog():
    start_node = simpledialog.askstring("Input", "Enter the start node:")
    end_node = simpledialog.askstring("Input", "Enter the end node:")
    speed = max(0.1, min(1.0, 1.0 - (int(speed_var.get()) - 1) / 9))

    if start_node and end_node:
        start_node = start_node.upper()
        end_node = end_node.upper()

        # Call the branch and bound search
        order = branch_and_bound(G, start_node, end_node)

        # Display the visualization
        title = 'Branch and Bound Visualization'
        visualization(order, title, G, pos, speed)

def visualize_a_star_dialog():
    start_node = simpledialog.askstring("Input", "Enter the start node:")
    end_node = simpledialog.askstring("Input", "Enter the end node:")
    speed = max(0.1, min(1.0, 1.0 - (int(speed_var.get()) - 1) / 9))

    if start_node and end_node:
        start_node = start_node.upper()
        end_node = end_node.upper()

        # Call the A* search with the heuristic function
        order = a_star(G, start_node, end_node, my_heuristic)

        # Display the visualization
        title = 'A* Search Visualization'
        visualization_with_heuristic(order, title, G, pos, my_heuristic, end_node, speed)

def generate_random_graph_dialog():
    try:
        num_vertices = simpledialog.askinteger("Input", "Enter the number of vertices:")
        num_edges = simpledialog.askinteger("Input", "Enter the number of edges:")
        
        if num_vertices is not None and num_edges is not None:
            G.clear()
            G.update(generate_random_graph(num_vertices, num_edges))
            update_graph()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def generate_random_connected_graph_dialog():
    try:
        num_vertices = simpledialog.askinteger("Input", "Enter the number of vertices:")
        if num_vertices is not None:
            G.clear()
            G.update(generate_random_connected_graph(num_vertices))
            update_graph()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def add_vertex():
    try:
        vertex = entry_vertex.get().upper()
        if not vertex.isalpha() or len(vertex) != 1:
            raise ValueError("Please enter a valid single-letter vertex.")
        
        G.add_node(vertex)
        update_graph()
        entry_vertex.delete(0, 'end')  # Clear the entry widget
    except Exception as e:
        messagebox.showerror("Error", str(e))

def delete_vertex():
    try:
        vertex = entry_vertex.get().upper()
        if not vertex.isalpha() or len(vertex) != 1:
            raise ValueError("Please enter a valid single-letter vertex.")

        G.remove_node(vertex)
        update_graph()
        entry_vertex.delete(0, 'end')  # Clear the entry widget
    except Exception as e:
        messagebox.showerror("Error", str(e))        
        
def add_edge():
    try:
        # Get the input from the entry widget and convert it to uppercase
        edge_input = entry_edge.get().upper()
        
        # Check if the input has at least 3 characters and follows the specified format
        if len(edge_input) < 3 or not edge_input[:2].isalpha() or not edge_input[2:].isdigit():
            raise ValueError("Please enter a valid input. Format: 'node+node+weight'")
        
        # Extract the first two characters as vertices and the remaining characters as weight
        vertices = edge_input[:2]
        weight = int(edge_input[2:])

        # Add an edge to the graph with the specified vertices and weight
        G.add_edge(vertices[0], vertices[1], weight=weight)
        
        # Update and redraw the graph
        update_graph()
        
        # Clear the entry widget
        entry_edge.delete(0, 'end')

    # Handle the case where a ValueError is raised (e.g., invalid input format)
    except ValueError as ve:
        messagebox.showerror("Error", str(ve))

    # Handle other exceptions and show an error message
    except Exception as e:
        messagebox.showerror("Error", str(e))


def update_graph():
    global pos
    pos = nx.spring_layout(G)

    plt.clf()
    plt.title("Graph Creator")
    edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
    nx.draw(G, pos, with_labels=True, node_color='lightblue')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.show()

def on_close():
    root.destroy()

def pause_visualization():
    global visualization_paused
    visualization_paused = True

def play_visualization():
    global visualization_paused
    visualization_paused = False

# Create the main window
root = Tk()
root.title("Searching Algorithm Visualizer Menu")

# Create a graph
G = nx.Graph()
pos = nx.spring_layout(G)

# Create GUI components
label_vertex = Label(root, text="Add Vertex (single letter):")
entry_vertex = Entry(root, width=10)
button_add_vertex = Button(root, text="Add Vertex", command=add_vertex)
button_delete_vertex = Button(root, text="Delete Vertex", command=delete_vertex)


label_edge = Label(root, text="Add Edge (e.g., 'AB5'):")
entry_edge = Entry(root, width=10)
button_add_edge = Button(root, text="Add Edge", command=add_edge)
button_delete_vertex.grid(row=0, column=3, padx=5, pady=5)

label_weight = Label(root, text="Edge Weight:")
entry_weight = Entry(root, width=10)

label_speed = Label(root, text="Animation Speed (1-10):")
speed_var = StringVar()
speed_var.set("5")  # Default animation speed
speed = Entry(root, textvariable=speed_var, width=10)

button_visualize_dfs = Button(root, text="Visualize DFS", command=lambda: visualize_dfs_dialog())
button_visualize_bfs = Button(root, text="Visualize BFS", command=lambda: visualize_bfs_dialog())
button_visualize_hill_climbing = Button(root, text="Visualize Hill Climbing", command=visualize_hill_climbing_dialog)
button_visualize_beam_search = Button(root, text="Visualize Beam Search", command=visualize_beam_search_dialog)
button_visualize_branch_and_bound = Button(root, text="Visualize Branch and Bound", command=visualize_branch_and_bound_dialog)
button_visualize_a_star = Button(root, text="Visualize A*", command=visualize_a_star_dialog)
button_generate_random_graph = Button(root, text="Generate Random Graph", command=generate_random_graph_dialog)
button_generate_random_connected_graph = Button(root, text="Generate Random Connected Graph", command=generate_random_connected_graph_dialog)
button_close = Button(root, text="Close", command=on_close)

# Play and pause buttons
button_pause = Button(root, text="Pause", command=lambda: pause_visualization())
button_play = Button(root, text="Play", command=lambda: play_visualization())

# Arrange GUI components
label_vertex.grid(row=0, column=0, padx=5, pady=5)
entry_vertex.grid(row=0, column=1, padx=5, pady=5)
button_add_vertex.grid(row=0, column=2, padx=5, pady=5)

label_edge.grid(row=1, column=0, padx=5, pady=5)
entry_edge.grid(row=1, column=1, padx=5, pady=5)
button_add_edge.grid(row=1, column=2, padx=5, pady=5)

label_weight.grid(row=2, column=0, padx=5, pady=5)
entry_weight.grid(row=2, column=1, padx=5, pady=5)

label_speed.grid(row=4, column=2, padx=5, pady=5)
speed.grid(row=4, column=3, padx=5, pady=5)

button_visualize_dfs.grid(row=2, column=0, padx=5, pady=5)
button_visualize_bfs.grid(row=2, column=1, padx=5, pady=5)
button_visualize_hill_climbing.grid(row=2, column=2, padx=5, pady=5)
button_visualize_beam_search.grid(row=2, column=3, padx=5, pady=5)
button_visualize_branch_and_bound.grid(row=3, column=0, padx=5, pady=5)
button_visualize_a_star.grid(row=3, column=1, padx=5, pady=5)
button_generate_random_graph.grid(row=4, column=0, padx=5, pady=5)
button_generate_random_connected_graph.grid(row=4, column=1, padx=5, pady=5)

button_pause.grid(row=3, column=2, padx=5, pady=5)
button_play.grid(row=3, column=3, padx=5, pady=5)
button_close.grid(row=3, column=5, padx=5, pady=5)

# Update and display the initial graph
update_graph()

# Start the GUI main loop
root.mainloop()