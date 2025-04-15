import numpy as np
from scipy.special import comb
# import matplotlib.pyplot as plt
# import casadi as ca
import aerosandbox as asb
from scipy.optimize import minimize

#cosine points distribution
N_POINTS = 2000
THETA = np.linspace(0, np.pi, N_POINTS, endpoint=True)
X = 0.5 * (1 - np.cos(THETA))

#CST coefficients
N_1 = 0.5
N_2 = 1
#Number of CST weights
N_WEIGHTS = 8

#Freestream parameters
V = 3
RHO = 1.225
MU = 0.0000181
C = 1

#Dimensionless parameters
RE = RHO * V * C / MU
MACH = V / 343



#Boundaries for constraints
CM_BOUND = 0.15
CD_BOUND = 0.02

def bernstein_poly(n, i, x):
    return comb(n, i) * (x**i) * ((1 - x)**(n - i))

def class_function(x):
    return (x**N_1) * ((1 - x)**N_2)

def shape_function(x, w):
    n = len(w) - 1
    S = 0
    for i in range(n + 1):
        S += w[i] * bernstein_poly(n, i, x)
    return S

def CST(x, w):
    return class_function(x) * shape_function(x, w)

def generate_airfoil(w):
    w_upper = w[:N_WEIGHTS]
    w_lower = w[N_WEIGHTS:]

    y_upper = CST(X, w_upper)
    y_lower = CST(X, w_lower)

    x_airfoil = np.concatenate([X[::-1], X[1:]])
    y_airfoil = np.concatenate([y_upper[::-1], y_lower[1:]])

    # y_camber = (y_upper + y_lower) / 2

    coords = np.vstack((x_airfoil, y_airfoil)).T
    af = asb.Airfoil(name='generated', coordinates=coords)

    return af

def thickness_at_x(x_query, x, y_upper, y_lower):
    i_closest = np.argmin(np.abs(x - x_query))
    return y_upper[i_closest] - y_lower[i_closest]

def generate_aerodata(af,alpha):
    data = af.get_aero_from_neuralfoil(alpha=alpha, Re=RE, mach=MACH)

    CL = data["CL"]
    CD = data["CD"]
    CM = data["CM"]

    return CL,CD, CL/CD if CD!=0 else 0.0, CM

def objective(w):
    af = generate_airfoil(w)
    angles =  [2,  4, 6, 8, 10, 12, 14, 16]
    weights = [10, 9, 8, 7, 6,   5,  4,  3]

    CLCD_total = 0

    for alpha, weight in zip(angles, weights):
        CL, CD, CLCD,_ = generate_aerodata(af, alpha)

        if CD <=0 or CL<=0:
            CLCD_total += 1e6
        else:
            CLCD_total += (weight/np.sum(weights)) * CLCD

    CLCD_total /= len(angles)

    return -CLCD_total

def constraint_LE_angle(w):
    af= generate_airfoil(w)

    upper_LE_vec = [af.x()[N_POINTS-2] - af.x()[N_POINTS-1], af.y()[N_POINTS-2] - af.y()[N_POINTS-1]]
    lower_LE_vec = [af.x()[N_POINTS] - af.x()[N_POINTS-1], af.y()[N_POINTS] - af.y()[N_POINTS-1]]

    dot = np.dot(upper_LE_vec,lower_LE_vec)
    l1 = np.linalg.norm(upper_LE_vec)
    l2 = np.linalg.norm(lower_LE_vec)

    # cos_angle = dot / (l1 * l2 + 1e-10)  # avoid divide-by-zero

    return dot + l1*l2

def constraint_TE_angle(w):
    af = generate_airfoil(w)

    upper_TE_vec = [af.x()[0]- af.x()[1],af.y()[0]-af.y()[1]]
    lower_TE_vec = [af.x()[-1]- af.x()[-2], af.y()[-1] - af.y()[-2]]

    TE_angle =np.rad2deg(np.arctan2(
            upper_TE_vec[0] * lower_TE_vec[1] - upper_TE_vec[1] * lower_TE_vec[0],
            upper_TE_vec[0] * lower_TE_vec[0] + upper_TE_vec[1] * lower_TE_vec[1]))

    return TE_angle - 6.25

def constraint_CM(w):
    af = generate_airfoil(w)

    angles =  [2,  4, 6, 8, 10, 12, 14, 16]
    weights = [10, 9, 8, 7, 6,   5,  4,  3]

    CM_total = 0

    for alpha, weight in zip(angles, weights):
        _,_,_, CM = generate_aerodata(af,alpha)
        CM_total += (weight/np.sum(weights)) * CM

    CM_total /= len(angles)


    return CM_total + CM_BOUND

def constraint_CD(w):
    af = generate_airfoil(w)



    angles =  [2,  4, 6, 8, 10, 12, 14, 16]
    weights = [10, 9, 8, 7, 6,   5,  4,  3]

    CD_total = 0

    for alpha, weight in zip(angles, weights):
        _,CD,_,_ = generate_aerodata(af,alpha)


        CD_total += (weight/np.sum(weights)) * CD

    CD_total /= len(angles)
    print(CD_total)

    return CD_total - CD_BOUND

def constraint_upper_weight(w):
    w_up = w[:N_WEIGHTS]
    return w_up[0] - 0.05

def constraint_lower_weight(w):
    w_lo = w[N_WEIGHTS:]
    return 0.05 - w_lo[0]

def constraint_thickness_33(w):
    w_up = w[:N_WEIGHTS]
    w_lo = w[N_WEIGHTS:]
    y_up = CST(X, w_up)
    y_lo = CST(X, w_lo)
    thick_33 = thickness_at_x(0.33, X, y_up, y_lo)
    return thick_33 - 0.128

def constraint_thickness_90(w):

    w_up = w[:N_WEIGHTS]
    w_lo = w[N_WEIGHTS:]
    y_up = CST(X, w_up)
    y_lo = CST(X, w_lo)
    thick_90 = thickness_at_x(0.90, X, y_up, y_lo)
    return thick_90 - 0.014

def constraint_coords(w):

    w_up = w[:N_WEIGHTS]
    w_lo = w[N_WEIGHTS:]

    y_up = CST(X, w_up)
    y_lo = CST(X, w_lo)

    return np.min(y_up - y_lo)

def get_wiggliness(w):
   return sum([np.sum(np.diff(np.diff(array)) ** 2) for array in [w[:N_WEIGHTS], w[N_WEIGHTS:]]])

def constraint_wiggliness(w):
    mh114 = asb.Airfoil('mh114').to_kulfan_airfoil().set_TE_thickness(0.01)

    w0_lower = list(mh114.kulfan_parameters['lower_weights'])
    w0_upper = list(mh114.kulfan_parameters['upper_weights'])

    # w0_upper = [0.17693258, 0.18318778, 0.22626059, 0.18959734, 0.20129803, 0.20030807, 0.20133783, 0.20852971]
    # w0_lower = [-0.17007317, -0.12075837, -0.1204467 , -0.0658519 , -0.12849652, -0.05225167, -0.09086747, -0.07038213]


    w0 = w0_upper + w0_lower
    # return -get_wiggliness(w) + 0.1*get_wiggliness(w0)
    return -get_wiggliness(w) + 0.2



#initial guess


naca2412 = asb.Airfoil('naca2412').to_kulfan_airfoil()

w0_lower = list(naca2412.kulfan_parameters['lower_weights'])
w0_upper = list(naca2412.kulfan_parameters['upper_weights'])


# w0_upper = [0.18668034, 0.17505264, 0.45980884, 0.21222233, 0.4411998 , 0.341926  , 0.34066697, 0.42619503]
# w0_lower = [-0.15563574, -0.20725402,  0.14707671, -0.14028172,  0.13198311, 0.04489351,  0.08685891,  0.25107169]

mh114 = asb.Airfoil('mh114').to_kulfan_airfoil()

w0_lower = list(mh114.kulfan_parameters['lower_weights'])
w0_upper = list(mh114.kulfan_parameters['upper_weights'])

# w0_upper = [0.17693258, 0.18318778, 0.22626059, 0.18959734, 0.20129803, 0.20030807, 0.20133783, 0.20852971]
# w0_lower = [-0.17007317, -0.12075837, -0.1204467 , -0.0658519 , -0.12849652, -0.05225167, -0.09086747, -0.07038213]


w0 = w0_upper + w0_lower






cons = [
    {'type': 'eq', 'fun': constraint_TE_angle},
    {'type': 'ineq', 'fun': constraint_LE_angle},
    {'type': 'ineq', 'fun': constraint_thickness_90},
    {'type': 'ineq', 'fun': constraint_thickness_33},
    {'type': 'ineq', 'fun': constraint_CM},
    # {'type': 'ineq', 'fun': constraint_coords},
    # {'type': 'ineq', 'fun': constraint_CD},
    {'type': 'ineq', 'fun': constraint_lower_weight},
    {'type': 'ineq', 'fun': constraint_upper_weight},
    {'type': 'ineq', 'fun': constraint_wiggliness},
]

# Optional: Bound each weight to avoid degenerate shapes
# bnds = [(-0.5, 0.5)] * len(w0)
bnds = [(-0.25, 0.5)] * N_WEIGHTS + [(-0.5, 0.25)] * N_WEIGHTS

# Run the optimizer using SLSQP
res = minimize(
    fun=objective,
    x0=w0,
    method='SLSQP',
    bounds=bnds,
    constraints=cons,
    options={'maxiter': 4000, 'disp': True}
)


if res.success:
    print("Optimization succeeded!")
else:
    print("Optimization failed!")
print("Optimal weights (upper):", res.x[:N_WEIGHTS])
print("Optimal weights (lower):", res.x[N_WEIGHTS:])
print("Objective function value ( -weighted CL/CD ):", res.fun)
print("=> Maximum weighted CL/CD is approximately:", -res.fun)


af = generate_airfoil(w0)
CL, CD, CLCD, CM = generate_aerodata(af,2)
print(f"Original data: CL = {CL}, CD = {CD}, CL/CD = {CLCD}, CM = {CM}")

# original_wiggliness = 0
# for array in [w0_lower, w0_upper]:
#     original_wiggliness +=np.sum(np.diff(np.diff(array)) ** 2)
# print(f"Original wiggliness: {original_wiggliness}")





# w_opt_upper = res.x[:N_WEIGHTS]
# w_opt_lower = res.x[N_WEIGHTS:]


final_af = generate_airfoil(res.x)
CL, CD, CLCD, CM = generate_aerodata(final_af,2)

print(f"Optimized data: CL = {CL}, CD = {CD}, CL/CD = {CLCD}, CM = {CM}")
print(get_wiggliness(res.x))
# optimized_wiggliness = 0
# for array in [res.x[:N_WEIGHTS], res.x[N_WEIGHTS:]]:
#     optimized_wiggliness +=np.diff(np.diff(array)) ** 2
# print(f"Optimized wiggliness: {optimized_wiggliness}")
# print("Reynolds number:", RE, "Mach:", MACH)

final_af.draw()
# af.draw()
