"""

read me file - expalantion of the code 


This project presents a simulation environment for analysing the energy use and routing behaviour of a small
electric delivery drone. The goal is to model how such a drone performs multiple delivery trips in a simplified
urban setting, taking into account realistic flight energetics, grid-based navigation, and battery constraints.

The city is represented as a 10×10 grid. A warehouse, several delivery points, and a few no-fly cells are placed
randomly within this space. Each delivery point has an associated payload mass that the drone must carry from 
the warehouse. The drone can move in eight directions, and distances are calculated using standard geometric 
step lengths. Through repeated trips—each beginning with a full battery recharge—the drone attempts to service 
all feasible delivery locations.

A key component of the project is its physically motivated energy model. Takeoff and landing are simulated 
using ordinary differential equations that describe thrust, drag, induced velocity, and electrical power 
draw. These equations are integrated numerically using SciPy’s solve_ivp, allowing the system to determine 
the energy required to climb to a target altitude or descend safely. Horizontal cruise energy is computed 
using a combination of an ODE-based flight model and an analytical power model when needed. Together, these 
components provide realistic estimates of the energy required for takeoff, cruise, and landing for different 
payload masses.

To support the analysis, the project includes custom numerical tools. A simple one-dimensional linear 
interpolator is used to smooth ODE outputs such as altitude and battery state. A home-built root-finding 
routine—combining incremental search with bisection—is used to determine quantities such as the time when 
a specific altitude is reached or the speed at which induced and drag power become equal. These implementations 
help demonstrate numerical methods in a clear and accessible way.

Navigation and routing are handled using Dijkstra’s algorithm to precompute shortest paths between all important 
points on the grid. The drone’s trip-planning logic then uses these distances, together with the energy models, 
to decide which deliveries are possible with the remaining battery. The planner always ensures the drone can 
safely return to the warehouse and land before initiating a new trip. If a delivery would exceed the available 
energy margin, it is deferred to a later trip.

The project also provides several visualisation tools. These include plots of the takeoff altitude profile, 
battery energy consumption, range and endurance as functions of payload, and total deliverable mass per battery 
cycle. The grid is displayed graphically with the drone’s paths coloured according to battery percentage, making 
it easy to assess where energy is being spent. A continuous, interpolated trajectory plot demonstrates how the 
discrete grid movements translate into a smooth flight path over time.

In summary, this project combines flight physics, numerical computation, and routing algorithms into a coherent 
simulation suitable for engineering analysis. It offers a practical platform for exploring energy limitations, 
operational strategies, and design considerations in small unmanned aerial delivery systems, and serves as a 
solid basis for further development or research.


"""