# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors

# constants
G = 6.67430e-11              # gravitational constant
dt = 60*60*24                # time step (s)
tolerance = 1e-6             # softening length (m)
scale = 5e12                 # scale of plot (m)
diagnostics_factor = 1e-21   # switches to peta to avoid overflow issues

# Body class
class Body:
    def __init__(self, mass, radius, rvector, vvector):
        self.mass = float(mass) # mass (kg)
        self.radius = float(radius) # radius of body (m)
        self.rvector = np.array(rvector, dtype=np.float64) # position vector (m, m, m)
        self.vvector = np.array(vvector, dtype=np.float64) # velocity vector (m/s, m/s, m/s)

    # acceleration due to another body
    def acceleration_due_to(self, other):
        r = other.rvector - self.rvector
        dist = np.linalg.norm(r)

        # avoid exact zero divide
        if dist == 0.0:
            return np.zeros(3, dtype=np.float64)

        d = max(dist, tolerance)
        # acceleration: G * m_other * r / d^3
        a = G * other.mass * r / (d**3)

        return a

    # kinetic energy method
    def get_kinetic_energy(self):
        return 0.5 * self.mass * np.dot(self.vvector, self.vvector) * diagnostics_factor

    # momentum method
    def get_momentum(self):
        return self.mass * self.vvector * diagnostics_factor

    # angular mommentum method (with origin of r as reference point)
    def get_angular_momentum(self):
        return np.cross(self.rvector, self.get_momentum()) * diagnostics_factor

    # check if this body collides with another
    def collision_check(self, other):
        r = self.rvector - other.rvector
        dist = np.linalg.norm(r)
        a = self.radius + other.radius
        return dist <= a
    
    # elastic collision between two bodies
    def elastic_collision(self, other):
        r = self.rvector - other.rvector
        dist = np.linalg.norm(r)
        if dist == 0:
            r = np.array([1e-6, 0, 0], dtype=np.float64)
            dist = 1e-6
            self.rvector += r
        n = r / dist

        vn = np.dot(self.vvector - other.vvector, n)
        if vn >= 0:
            return  # moving apart

        m1, m2 = self.mass, other.mass
        # Standard 1D collision along normal
        self.vvector -= 2 * m2 / (m1 + m2) * vn * n
        other.vvector += 2 * m1 / (m1 + m2) * vn * n

# energy helper function
# returns energy of a lsit of bodies in J
def energy(bodies):
    T = 0.0
    U = 0.0
    n = len(bodies)
    for i in range(n):
        T += bodies[i].get_kinetic_energy()
        for j in range(i+1, n):
            r = bodies[j].rvector - bodies[i].rvector
            dist = np.linalg.norm(r)
            d = max(dist, tolerance) # avoids /0 errors
            U += -G * bodies[i].mass * bodies[j].mass / d * diagnostics_factor
    E = T + U
    return E

# momentum helper function
# returns momentum of a lsit of bodies
def momentum(bodies):
    p = np.zeros(3, dtype=np.float64)
    for b in bodies:
        p += b.get_momentum()
    return p

# angular momentum helper function
# returns angular momentum of a list of bodies
def angular_momentum(bodies):
    L = np.zeros(3, dtype=np.float64)
    for b in bodies:
        L += b.get_angular_momentum()
    return L

# compute accelerations for a list of bodies
def interact(bodies):
    n = len(bodies)
    accs = [np.zeros(3, dtype=np.float64) for _ in range(n)]
    for i, b1 in enumerate(bodies):
        a = np.zeros(3, dtype=np.float64)
        for j, b2 in enumerate(bodies):
            if i == j:
                continue
            a += b1.acceleration_due_to(b2)
        accs[i] = a
    return accs

# steps through calculating new positions and velocities of
# the different bodies, using leapfrog integration
def step(bodies):
    # leapfrog method of calculation
    accs = interact(bodies)

    # half-step the velocity
    for i, b in enumerate(bodies):
        b.vvector += + accs[i] * (dt * 0.5)

    # update position
    for b in bodies:
        b.rvector += + b.vvector * dt

    # update accelerations
    accs2 = interact(bodies)

    # half-step velocity again
    for i, b in enumerate(bodies):
        b.vvector += + accs2[i] * (dt * 0.5)

    # collisions
    n = len(bodies)
    for i in range(n):
        for j in range(i+1, n):
            if bodies[i].collision_check(bodies[j]):
                print("Collision!")
                bodies[i].elastic_collision(bodies[j])
                

# Bodies generator
'''
IMPORTANT
Set the N value to the amount of bodies you wish to simulate
Even if you are using your own body lsit, and not randomly generated bodies
This tells the plot how many colros to generate for them

If you're using the random seed method, this tells it how many bodies to generate
'''
N = 9
colors = plt.cm.tab10(np.arange(N) % 10)  # tab10 has 10 distinct colors

# solar system parameters

# rotation matrix helper, angles in radians
rotation_matrix = lambda theta: np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])

mass_sun = 2e30
radius_sun = 7e8
r_sun = np.array([0,0,0])
v_sun = np.array([0,0,0])

mass_mercury = 3.3e23
radius_mercury = 2.44e6
theta_mercury = np.radians(167)
r_mercury = rotation_matrix(theta_mercury) @ np.array([5.8e10, 0 ,0])
v_mercury = rotation_matrix(theta_mercury) @ np.array([0, 4.8e4, 0])

mass_venus = 4.9e24
radius_venus = 6.1e6
theta_venus = np.radians(15)
r_venus = rotation_matrix(theta_venus) @ np.array([1.1e11, 0, 0])
v_venus = rotation_matrix(theta_venus) @ np.array([0, 3.5e4, 0])

mass_earth = 6e24
radius_earth = 6e6
r_earth = np.array([1.5e11,0,0])
v_earth=np.array([0,3e4,0])

mass_mars = 6.4e23
radius_mars = 3.4e6
theta_mars = np.radians(8.5)
r_mars = rotation_matrix(theta_mars) @ np.array([2.3e11, 0, 0])
v_mars = rotation_matrix(theta_mars) @ np.array([0, 2.4e4, 0])

mass_jupiter = 1.9e27
radius_jupiter = 7e7
theta_jupiter = 0.162
r_jupiter = rotation_matrix(theta_jupiter) @ np.array([7.8e11, 0, 0])
v_jupiter = rotation_matrix(theta_jupiter) @ np.array([0, 1.3e4, 0])

mass_saturn = 5.7e26
radius_saturn = 5.8e7
theta_saturn = np.radians(5.4)
r_saturn = rotation_matrix(theta_saturn) @ np.array([1.4e12, 0, 0])
v_saturn = rotation_matrix(theta_saturn) @ np.array([0, 9.7e3, 0])

mass_uranus = 8.7e25
radius_uranus = 2.6e7
theta_uranus = np.radians(1.6)
r_uranus = rotation_matrix(theta_uranus) @ np.array([2.9e12, 0, 0])
v_uranus = rotation_matrix(theta_uranus) @ np.array([0, 6.8e3, 0])

mass_neptune = 1e26
radius_neptune = 2.5e7
theta_neptune = np.radians(1.4)
r_neptune = rotation_matrix(theta_neptune) @ np.array([4.5e12, 0, 0])
v_neptune = rotation_matrix(theta_neptune) @ np.array([0, 5.4e3, 0])

# lgrange points
mass_lagrange = 1
radius_lagrange = 1

r_lag1 = (1 - (mass_earth / (3 * (mass_earth + mass_sun)))**(1/3)) * r_earth
v_lag1 = v_earth

r_lag2 = (1 + (mass_earth / (3 * (mass_earth + mass_sun)))**(1/3)) * r_earth
v_lag2 = v_earth

r_lag3 = -1 * r_earth
v_lag3 = -1 * v_earth

r_lag4 = rotation_matrix(np.pi / 3) @ r_earth
v_lag4 = rotation_matrix(np.pi / 3) @ v_earth

r_lag5 = rotation_matrix(-np.pi / 3) @ r_earth
v_lag5 = rotation_matrix(-np.pi / 3) @ v_earth

bodies = [
    Body(mass_sun, radius_sun, r_sun, v_sun),
    Body(mass_mercury, radius_mercury, r_mercury, v_mercury),
    Body(mass_venus, radius_venus, r_venus, v_venus),
    Body(mass_earth, radius_earth, r_earth, v_earth),
    Body(mass_mars, radius_mars, r_mars, v_mars),
    Body(mass_jupiter, radius_jupiter, r_jupiter, v_jupiter),
    Body(mass_saturn, radius_saturn, r_saturn, v_saturn),
    Body(mass_uranus, radius_uranus, r_uranus, v_uranus),
    Body(mass_neptune, radius_neptune, r_neptune, v_neptune),
    # Body(mass_lagrange, radius_lagrange, r_lag1, v_lag1),
    # Body(mass_lagrange, radius_lagrange, r_lag2, v_lag2),
    # Body(mass_lagrange, radius_lagrange, r_lag3, v_lag3),
    # Body(mass_lagrange, radius_lagrange, r_lag4, v_lag4),
    # Body(mass_lagrange, radius_lagrange, r_lag5, v_lag5)
]

for i, body in enumerate(bodies):
    print(f"Body{i}: mass: {body.mass / mass_earth:.1e} (Earth masses), distance from origin: {np.linalg.norm(body.rvector) / np.linalg.norm(r_earth):.1e} (AU), phase: {np.arccos(np.dot(body.rvector, np.array([1,0,0]) / np.linalg.norm(body.rvector))):.3e} (rad) velocity: {np.linalg.norm(body.vvector) / np.linalg.norm(v_earth):.1e} (Earth Velocities)")

''''
Uncomment this, and comment out the bodies[] line to use random bodies
'''
# for n in range(N):
#     m = np.random.uniform(1e24, 1e27)
#     r = np.random.uniform(1, 10)
#     theta = 2*np.pi*n/N
#     pos = np.random.uniform(-2.5e11, 2.5e11, 3)
#     pos[2] = 0
#     vel = np.cross(pos, np.array([0,0,3e-8]))
#     # vel = np.random.uniform(-1e2, 2e2, 3)
#     bodies.append(Body(m, r, pos, vel))

# initial diagnostics
E0 = energy(bodies)
P0 = momentum(bodies)
L0 = angular_momentum(bodies)
print(f"Initial E: {E0:.6e}, |P|: {np.linalg.norm(P0):.6e}, |L|: {np.linalg.norm(L0):.6e}")

# plotting
fig, ax_sim = plt.subplots(1,1, figsize=(7,10))
ax_sim.set_xlim(-scale, scale)
ax_sim.set_ylim(-scale, scale)
ax_sim.set_aspect('equal')
scatter = ax_sim.scatter([b.rvector[0] for b in bodies], [b.rvector[1] for b in bodies],
                         s=[np.clip(np.log10(b.mass), 1, 50) for b in bodies],
                         c=colors)

frames = []
steps = 0

# update each frame
def update(frame):
    global steps
    step(bodies)
    xs = [b.rvector[0] for b in bodies]
    ys = [b.rvector[1] for b in bodies]
    scatter.set_offsets(np.c_[xs, ys])

    frames.append(frame)

    steps += 1
    return scatter

# Run animation
ani = FuncAnimation(fig, update, frames=3599, interval=10, blit=False, repeat=True) # change repeat to false for data
# change frames to fit how many dts you want to simulate, currently set to 3599+1 = 3600 dts (dt in day, typically)
plt.show()

# final diagnostics
print(f"Final time: {steps * dt / 60 / 60 / 24} Days")
for i, body in enumerate(bodies):
    print(f"Body{i}: mass: {body.mass / mass_earth:.1e} (Earth masses), distance from origin: {np.linalg.norm(body.rvector) / np.linalg.norm(r_earth):.1e} (AU), phase: {np.arccos(np.dot(body.rvector, np.array([1,0,0]) / np.linalg.norm(body.rvector))):.3e} (rad) velocity: {np.linalg.norm(body.vvector) / np.linalg.norm(v_earth):.1e} (Earth Velocities)")
E_final = energy(bodies)
P_final = momentum(bodies)
L_final = angular_momentum(bodies)
print(f"Final: E={E_final:.3e}, |P|={np.linalg.norm(P_final):.3e}, |L|={np.linalg.norm(L_final):.3e}")
print(f"Energy error: {np.abs(E_final-E0)/E0:.3e}, |P| error: {np.abs(np.linalg.norm(P_final)-np.linalg.norm(P0))/np.linalg.norm(P0):.3e}, |L| error: {np.abs(np.linalg.norm(L_final)-np.linalg.norm(L0))/np.linalg.norm(L0):.3e}")