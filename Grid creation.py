#code to generat00 e the stops at a random location on the grid and assining weight values 
import numpy as np 
import random
grid = []
delivered_grid = []

grid_size = 10

def init_grid(): # deffinging the grid vaiable 
    for i in range(grid_size):
        row = [0 for i in range(grid_size)]
        grid.append(row)        
        
        delivered_row = [True for i in range(grid_size)]
        delivered_grid.append(delivered_row)
        

#adding number to a radom palce in the grid
def add_random_num():
    global grid
    zero_cells = []
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i][j] == 0:
                zero_cells.append((i, j))
    if len(zero_cells) > 0:
        random_index = random.randint(0, len(zero_cells) - 1)
        random_cell = zero_cells[random_index]
        weight = round(random.uniform(0.2, 2), 3)
        grid[random_cell[0]][random_cell[1]] = weight
        delivered_grid[random_cell[0]][random_cell[1]] = False 
        value = [random_cell, weight]
        return value

def adding_blocks():
    global grid
    zero_cells = []
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i][j] == 0:
                zero_cells.append((i, j))
    if len(zero_cells) > 0:
        random_index = random.randint(0, len(zero_cells) - 1)
        random_cell = zero_cells[random_index]
        grid[random_cell[0]][random_cell[1]] = "no_go"
        block = [random_cell]
        block
        return block
        
        
def adding_warhouses():
    global grid
    zero_cells = []
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i][j] == 0:
                zero_cells.append((i, j))
    if len(zero_cells) > 0:
        random_index = random.randint(0, len(zero_cells) - 1)
        random_cell = zero_cells[random_index]
        grid[random_cell[0]][random_cell[1]] = "warehouse"
        warehouse_location = random_cell
        return warehouse_location
        
def distance_to_warehouse(warehouse_location, values): 
    distance = [] 
    print(warehouse_location)
    for i in range(len(values)): 
        temp_distance = round(np.sqrt(values[i][0][0] - warehouse_location[0]**2) + (values[i][0][1] - warehouse_location[1])**2)
        temp = [values[0][0], temp_distance]
        distance.append(temp_distance)
        
    print(distance)
        
        
    

def main():  
    
    block_location = []
    value_locations = []
    init_grid()
    for i in range(7):
        value = add_random_num()
        value_locations.append(value)
        
    for i in range(4): 
        block = adding_blocks()
        block_location.append(block)
        
    warehouse_location = adding_warhouses() 
    
    for i in range(grid_size):     
        print(grid[i])
        
    for i in range(grid_size): 
        print(delivered_grid[i]) 
        
    distance_to_warehouse(warehouse_location, value_locations)

main()



