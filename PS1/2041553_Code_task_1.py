graph = {
    '0': [['1', '3'], 1],
    '1': [['0', '2', '5'], 1],
    '2': [['1', '5'], 1],
    '3': [['0', '4', '6'], 1],
    '4': [['1', '3', '5', '7'], 0],
    '5': [['2', '4', '8'], 1],
    '6': [['3', '7'], 1],
    '7': [['4', '6', '8'], 1],
    '8': [['5', '7'], 1]
}

visited = []  # List for visited nodes.
queue = []  # Initialize a queue


def bfs(visited, graph, node):  # function for BFS
    visited.append(node)

    # need to add dirty/clean flag
    queue.append(node)

    while queue:  # Creating loop to visit each node
        graph[node][1] = 0
        m = queue.pop(0)
        print(m, "clean", end=" ")

        for neighbour in graph[m][0]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)


print("Breadth-First Search Implementation")
bfs(visited, graph, '4')
