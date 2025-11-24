# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors

# constants
G = 6.67430e-11       # gravitational constant
dt = 60*60            # time step (s)
tolerance = 1e-1           # softening length (m)
scale = 3e11               # scale of plot (m)
diagnostics_factor = 1e-21 # switches to peta

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

# elastic collision between two bodies
def elastic_collision(b1, b2):
    r = b1.rvector - b2.rvector
    dist = np.linalg.norm(r)
    if dist == 0:
        r = np.array([1e-6, 0, 0], dtype=np.float64)
        dist = 1e-6
    n = r / dist

    vn = np.dot(b1.vvector - b2.vvector, n)
    if vn >= 0:
        return  # moving apart

    m1, m2 = b1.mass, b2.mass
    # Standard 1D collision along normal
    b1.vvector -= 2 * m2 / (m1 + m2) * vn * n
    b2.vvector += 2 * m1 / (m1 + m2) * vn * n

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
                elastic_collision(bodies[i], bodies[j])
                

# Bodies generator
N = 100
colors = plt.cm.tab10(np.arange(N) % 10)  # tab10 has 10 distinct colors
bodies = []

for n in range(N):
    m = np.random.uniform(1e24, 1e27)
    r = np.random.uniform(1, 10)
    theta = 2*np.pi*n/N
    pos = np.random.uniform(-2.5e11, 2.5e11, 3)
    pos[2] = 0
    vel = np.cross(pos, np.array([0,0,7e-8]))
    # vel = np.random.uniform(-1e2, 2e2, 3)
    bodies.append(Body(m, r, pos, vel))

# initial diagnostics
E0 = energy(bodies)
P0 = momentum(bodies)
L0 = angular_momentum(bodies)
print(f"Initial E: {E0:.6e}, |P|: {np.linalg.norm(P0):.6e}, |L|: {np.linalg.norm(L0):.6e}")

# plotting
fig, (ax_sim, ax_E, ax_P, ax_L) = plt.subplots(4,1, figsize=(7,10))
ax_sim.set_xlim(-scale, scale)
ax_sim.set_ylim(-scale, scale)
ax_sim.set_aspect('equal')
scatter = ax_sim.scatter([b.rvector[0] for b in bodies], [b.rvector[1] for b in bodies],
                         s=[np.clip(np.log10(b.mass), 1, 50) for b in bodies],
                         c=colors)

frames, energies, momenta, angulas = [], [], [], []
line_E, = ax_E.plot([], [], color='blue')
line_P, = ax_P.plot([], [], color='red')
line_L, = ax_L.plot([], [], color='green')
ax_E.set_title("Energy"); ax_P.set_title("Momentum"); ax_L.set_title("Angular Momentum")

# update each frame
def update(frame):
    step(bodies)
    xs = [b.rvector[0] for b in bodies]
    ys = [b.rvector[1] for b in bodies]
    scatter.set_offsets(np.c_[xs, ys])

    E = energy(bodies)
    P = np.linalg.norm(momentum(bodies))
    L = np.linalg.norm(angular_momentum(bodies))

    frames.append(frame); energies.append(E); momenta.append(P); angulas.append(L)

    line_E.set_data(frames, energies)
    line_P.set_data(frames, momenta)
    line_L.set_data(frames, angulas)

    ax_E.relim(); ax_E.autoscale_view()
    ax_P.relim(); ax_P.autoscale_view()
    ax_L.relim(); ax_L.autoscale_view()

    return scatter, line_E, line_P, line_L

# Run animation
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
plt.show()

# final diagnostics
E_final = energy(bodies)
P_final = momentum(bodies)
L_final = angular_momentum(bodies)
print(f"Final: E={E_final:.3e}, |P|={np.linalg.norm(P_final):.3e}, |L|={np.linalg.norm(L_final):.3e}")
print(f"Energy error: {np.abs(E_final-E0)/E0:.3e}, |P| error: {np.abs(np.linalg.norm(P_final)-np.linalg.norm(P0))/np.linalg.norm(P0):.3e}, |L| error: {np.abs(np.linalg.norm(L_final)-np.linalg.norm(L0))/np.linalg.norm(L0):.3e}")