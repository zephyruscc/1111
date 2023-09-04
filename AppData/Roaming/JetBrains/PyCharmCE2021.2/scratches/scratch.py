def solve_n_equation_recursive(n, target_sum, coefficients, current_solution, solutions):
    if n == 0:
        if target_sum == 0:
            solutions.append(current_solution.copy())
        return

    for i in range(target_sum // coefficients[n - 1] + 1):
        current_solution[n - 1] = i
        solve_n_equation_recursive(n - 1, target_sum - i * coefficients[n - 1], coefficients, current_solution,
                                   solutions)

def calculate_sequence(n):
    a = [0] * n
    a[0]=1
    for i in range(1,n):
        solutions = []
        solve_n_equation_recursive(i+1, i+1, list(range(1, i + 2)), [0] * (i+1), solutions)
        sum=0
        for solution in solutions:
            # print(solution)
            def calculate_exponential_product(i,arr):
                result = 1
                for j, num in zip(range(i), arr[1:]):
                    result *= a[j] ** num
                return result
            sum+=calculate_exponential_product(i,solution)
        a[i]=sum
        print(sum)
    return a

sequence = calculate_sequence(50)
print(sequence)

