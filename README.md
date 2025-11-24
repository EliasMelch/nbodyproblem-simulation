This contains the simulation code for the nbody problem used in my final paper for Physics 235W.
To configure the timestep, softening length, scale of plots, and scaling factor for diagnostics, edit the numbers at the top of the file
Currently, the file is set to simulate the solar system alongside the Lagrange points
To alter the bodies, or Lagrange point data, simply edit the section headed by solar system parameters
Make sure to update the N value above this for the correct number of bodies (this ensures the correct number of colors are generated for the plot)
For random bodies, uncomment the section below the solar system parameters, and comment out the section that generates Bodies[] after the soalr system section
Adjust the N value to taste for as many bodies as you want!
Bodies which fly off screen are still simulated, even if they aren't visible
Caution: simulating more bodies, especially with high mass, and especially with high velocity may break energy conservation
