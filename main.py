import gradio as gr
import numpy as np
from scipy.optimize import linprog

def solve_mctp(num_supply_centers, num_demand_centers, num_goods, supply_limits, demand_limits, transportation_costs):

    m = int(num_supply_centers)
    n = int(num_demand_centers)
    p = int(num_goods)

    supply_limits_input = [[float(value) for value in limits.split(',')] for limits in supply_limits.split(';')]
    demand_limits_input = [[float(value) for value in limits.split(',')] for limits in demand_limits.split(';')]
    transportation_costs_input = [
        [
            [float(value) for value in costs.split(',')]
            for costs in row.split(':')
        ]
        for row in transportation_costs.split(';')
    ]
    # Solver:
    c = []
    for i in range(m):
        for j in range(n):
            for k in range(p):
                c.append(transportation_costs_input[i][j][k])

    # Inequality constraints (supply constraints)
    A_ub = []
    b_ub = []
    for i in range(m):
        for k in range(p):
            zero = np.zeros((m, n, p))
            for j in range(n):
                zero[i][j][k] = 1
            zero = zero.reshape(-1).tolist()
            A_ub.append(zero)
            b_ub.append(supply_limits_input[i][k])

    # equality constraints:
    A_eq = []
    b_eq = []
    for j in range(n):
        for k in range(p):
            zero = np.zeros((m, n, p))
            for i in range(m):
                zero[i][j][k] = 1
            zero = zero.reshape(-1).tolist()
            A_eq.append(zero)
            b_eq.append(demand_limits_input[j][k])

    solver = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
    if solver.success:
        result = f"Optimal Value Found: {solver.success}\n"
        result += f"Optimal Value: ${solver.fun}\n"
        result += f"Optimal Solution: {solver.x}"
    else:
        result = "Solver did not find an optimal solution."

    return result


iface = gr.Interface(
    fn=solve_mctp,
    inputs=[
        gr.components.Number(label="Number of Supply Centers"),
        gr.components.Number(label="Number of Demand Centers"),
        gr.components.Number(label="Number of Goods"),
        gr.components.Textbox(label="Supply Limits (semicolon and comma-separated)"),
        gr.components.Textbox(label="Demand Limits (semicolon and comma-separated)"),
        gr.components.Textbox(label="Transportation Costs (semicolon, colon, and comma-separated)"),
    ],
    outputs=gr.outputs.Textbox(label="Solution"),
    title="Multi-Commodity Transportation Problem Solver",
    description="Enter problem data and click 'Solve' to find the solution.",
    examples=[
        [2, 2, 2, "10,15; 20,5", "15,10; 15,10", "2,3:3,2; 4,1:1,4"],
        [2, 2, 2, "50,60;70,80", "40,30;50,60", "2,3:3,2;4,1:1,4"],
        [2, 5, 1, "60000;60000", "30000;23000;15000;32000;16000", "1.75:2.25:1.5:2:1.5;2:2.5:2.5:1.5:1"]
    ]
)
if __name__ == "__main__":
    iface.launch()
