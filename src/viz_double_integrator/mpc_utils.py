from scipy.optimize import minimize
import numpy as np

dt = 0.1

def model(x, u) -> np.array:
    '''
        x: (2,1)
        u: (1,) or ()
    '''
    A = np.array(
        [[1, dt],
         [0, 1]]
    )
    B = np.array([[0], [dt]])
    x_next = A @ x + B * u
    assert x.shape == x_next.shape, print(x.shape, x_next.shape)
    return x_next

def cum_cost(x_init, u_seq, N):
    cost = 0.
    x = x_init
    assert u_seq.shape == (N,)
    for i in range(N):
        u = u_seq[i]  # shape = ()
        x_next = model(x, u)
        x = x_next
        cost = cost + np.linalg.norm(x, ord=1) + 0.05 * np.abs(u)
    
    return cost.item()

def ineq_cons_efr(x_init, u_seq, N):
    assert x_init.shape == (2, 1)
    u_seq = u_seq.squeeze()
    assert u_seq.shape == (N,), print(u_seq.shape)
    u_init = u_seq[0]
    x_next = model(x_init, u_init)
    x1, x2 = x_next.squeeze()
    ineq_list = [x1 + 5, 5 - x1, x2 + 5, 5 - x2]
    
    x2_max = np.sqrt(2 * (5 - x1) + 1e-4)
    x2_min = -np.sqrt(2 * (5 + x1) + 1e-4)

    ineq_list += [x2 - x2_min, x2_max - x2]
    return ineq_list

def ineq_cons_pointwise(x_init, u_seq, N, low, high):
    assert x_init.shape == (2, 1)
    u_seq = u_seq.squeeze()
    state_list = []
    x = x_init
    for i in range(N):
        u = u_seq[i]
        x_next = model(x, u)
        state_list += list(x_next.squeeze())
        x = x_next
    
    ineq_list = [state-low for state in state_list] + [high-state for state in state_list]
    return ineq_list

def get_action(x_init, N, cons_type='pointwise'):
    low, high = -5, 5
    x0 = np.ones((N, 1))
    bounds = [(-1., 1.)] * N
    if cons_type == 'pointwise':
        ineqcons = lambda u: ineq_cons_pointwise(x_init, u, N, low, high)
    elif cons_type == 'efr':
        ineqcons = lambda u: ineq_cons_efr(x_init, u, N)
    
    constraints = {'type': 'ineq', 'fun': ineqcons}
    res = minimize(
        lambda u: cum_cost(x_init, u, N), x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )

    return res.x, res.success

if __name__ == '__main__':
    np.set_printoptions(precision=4)
    horizon = 30

    x_init = np.array([[-4.8], [4]], dtype=np.float32)
    x_traj = [x_init.squeeze()]
    act_traj = []
    cost_traj = []
    x = x_init
    x1, x2 = x
    while abs(x1) > 0.1 or abs(x2) > 0.1:
        act, flag = get_action(x, horizon, cons_type='efr')
        x_next = model(x, act[0])
        x = x_next
        x1, x2 = x
        cost_traj.append(abs(x1) + abs(x2) + 0.05 * act[0])
        act_traj.append(act[0])
        x_traj.append(x.squeeze())
        print(x.squeeze())
    mpc_res = {
        'state': np.array(x_traj),
        'action': np.array(act_traj),
        'cost': np.array(cost_traj)
    }
    print(len(x_traj), sum(cost_traj))
    np.save('mpc.npy', mpc_res)