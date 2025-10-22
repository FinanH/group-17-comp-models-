#code to generate the stops at a random location on the grid and assining weight values 
import numpy as np 
import random
grid = []

def init_grid(): # deffinging the grid vaiable 
    for i in range(10):
        row = [0 for i in range(10)]
        grid.append(row)


#adding number to a radom palce in the grid
def add_random_num():
    global grid
    zero_cells = []
    for i in range(10):
        for j in range(10):
            if grid[i][j] == 0:
                zero_cells.append((i, j))
    if len(zero_cells) > 0:
        random_index = random.randint(0, len(zero_cells) - 1)
        random_cell = zero_cells[random_index]
        grid[random_cell[0]][random_cell[1]] = round(random.uniform(0.2, 2), 3)
        
init_grid()
for i in range(7):
    add_random_num() 

print(grid)