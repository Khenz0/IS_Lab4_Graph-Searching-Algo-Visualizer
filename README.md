# IS_Lab4_Graph-Searching-Algo-Visualizer
## Searching Algorithm Visualizer

This Python program provides a graphical interface to visualize various searching algorithms on a graph. The program uses the `networkx` library for graph representation, `matplotlib` for visualization, and `tkinter` for the GUI.

## Graph Representation and Generation

The graph is represented using the `networkx` library. The program includes functions to generate a random graph and a random connected graph.

## Searching Algorithms

The implemented searching algorithms include:

- Depth-First Search (DFS)
- Breadth-First Search (BFS)
- Hill Climbing
- Beam Search
- Branch and Bound
- A* Search

Each algorithm returns a sequence of states visited during the search process.

## Visualization

Graph visualization is done using the `matplotlib` library. Visualization functions are provided for DFS, BFS, Hill Climbing, Beam Search, Branch and Bound, and A* Search.

## How to Use

1. Run the program.
2. Use the GUI to add vertices, edges, and weights to the graph.
3. Visualize the different searching algorithms with or without heuristics.
4. Adjust the animation speed using the provided slider.
5. Pause and resume the visualization as needed.

## Dependencies

- `networkx`
- `matplotlib`
- `tkinter`

## Running the Program

Ensure you have the required dependencies installed. Run the program and use the GUI to interact with the graph and visualize the searching algorithms.

```bash
python your_program_filename.py
```
Feel free to explore and visualize different graphs and algorithms using the provided interface.
