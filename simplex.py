import ast, getopt, sys, copy, os
from fractions import Fraction

class SimplexSolver():
    def __init__(self):
        self.A = []
        self.b = []
        self.c = []
        self.tableau = []
        self.entering = []
        self.departing = []
        self.prob = "max"

    def run_simplex(self, A, b, c, prob = 'max'):
        self.prob = prob

        self.set_simplex_input(A, b, c)
            
        while (not self.should_terminate()):
            # Attempt to find a non-negative pivot.
            pivot = self.find_pivot()
            if pivot[1] < 0:
                print ("There exists no non-negative pivot. "
                        "Thus, the solution is infeasible.")
                return None

            # Do row operations to make every other element in column zero.
            self.pivot(pivot)

        solution = self.get_current_solution()
        print(("Current solution: %s\n" % str(solution)))
        
        return solution
        
    def set_simplex_input(self, A, b, c):
        # Convert all entries to fractions for readability.
        for a in A:
            self.A.append([Fraction(x) for x in a])    
        self.b = [Fraction(x) for x in b]
        self.c = [Fraction(x) for x in c]
            
        self.update_enter_depart(self.get_Ab())

        # If this is a minimization problem
        if self.prob == 'min':
            m = self.get_Ab()
            m.append(self.c + [0])
            m = [list(t) for t in zip(*m)] # Calculates the transpose
            self.A = [x[:(len(x)-1)] for x in m]
            self.b = [y[-1] for y in m]
            self.c = m[-1]
            self.A.pop()
            self.b.pop()
            self.c.pop()

        self.create_tableau()
        self.update_enter_depart(self.tableau)

    def update_enter_depart(self, matrix):
        self.entering = []
        self.departing = []
        # Create tables for entering and departing variables
        for i in range(0, len(matrix[0])):
            if i < len(self.A[0]):
                prefix = 'x' if self.prob == 'max' else 'y'
                self.entering.append("%s_%s" % (prefix, str(i + 1)))
            elif i < len(matrix[0]) - 1:
                self.entering.append("s_%s" % str(i + 1 - len(self.A[0])))
                self.departing.append("s_%s" % str(i + 1 - len(self.A[0])))
            else:
                self.entering.append("b")
    
    # Add slack & artificial variables to matrix A to transform
    #        all inequalities to equalities.    
    def add_slack_variables(self):
        slack_vars = self._generate_identity(len(self.tableau))
        for i in range(0, len(slack_vars)):
            self.tableau[i] += slack_vars[i]
            self.tableau[i] += [self.b[i]]

    # Create initial tableau table.        
    def create_tableau(self):
        self.tableau = copy.deepcopy(self.A)
        self.add_slack_variables()
        c = copy.deepcopy(self.c)
        for index, value in enumerate(c):
            c[index] = -value
        self.tableau.append(c + [0] * (len(self.b)+1))

    # Find pivot index.
    def find_pivot(self):
        enter_index = self.get_entering_var()
        depart_index = self.get_departing_var(enter_index)
        return [enter_index, depart_index]

    # Perform operations on pivot.
    def pivot(self, pivot_index):
        j, i = pivot_index

        pivot = self.tableau[i][j]
        self.tableau[i] = [element / pivot for
                           element in self.tableau[i]]
        for index, row in enumerate(self.tableau):
           if index != i:
              row_scale = [y * self.tableau[index][j]
                          for y in self.tableau[i]]
              self.tableau[index] = [x - y for x, y in
                                     zip(self.tableau[index],
                                         row_scale)]

        self.departing[i] = self.entering[j]
        
    # Get entering variable by determining the 'most negative'
    #        element of the bottom row.
    def get_entering_var(self):
        bottom_row = self.tableau[-1]
        most_neg_ind = 0
        most_neg = bottom_row[most_neg_ind]
        for index, value in enumerate(bottom_row):
            if value < most_neg:
                most_neg = value
                most_neg_ind = index
        return most_neg_ind
            
    # To calculate the departing variable, get the minimum of the ratio
    #        of b (b_i) to the corresponding value in the entering collumn. 
    def get_departing_var(self, entering_index):
        skip = 0
        min_ratio_index = -1
        min_ratio = 0
        for index, x in enumerate(self.tableau):
            if x[entering_index] != 0 and x[-1]/x[entering_index] > 0:
                skip = index
                min_ratio_index = index
                min_ratio = x[-1]/x[entering_index]
                break
        
        if min_ratio > 0:
            for index, x in enumerate(self.tableau):
                if index > skip and x[entering_index] > 0:
                    ratio = x[-1]/x[entering_index]
                    if min_ratio > ratio:
                        min_ratio = ratio
                        min_ratio_index = index
        
        return min_ratio_index

    # Get A matrix with b vector appended.
    def get_Ab(self):
        matrix = copy.deepcopy(self.A)
        for i in range(0, len(matrix)):
            matrix[i] += [self.b[i]]
        return matrix

    # Determines whether there are any negative elements
    #        on the bottom row
    def should_terminate(self):
        result = True
        index = len(self.tableau) - 1
        for i, x in enumerate(self.tableau[index]):
            if x < 0 and i != len(self.tableau[index]) - 1:
                result = False
        return result

    # Get the current solution from tableau.
    def get_current_solution(self):
        solution = {}
        for x in self.entering:
            if x != 'b':
                if x in self.departing:
                    solution[x] = self.tableau[self.departing.index(x)]\
                                  [len(self.tableau[self.departing.index(x)])-1]
                else:
                    solution[x] = 0
        solution['z'] = self.tableau[-1][len(self.tableau[0]) - 1]
        
        # If this is a minimization problem...
        if self.prob == 'min':
            bottom_row = self.tableau[-1]
            for v in self.entering:
                if 's' in v:
                    solution[v.replace('s', 'x')] = bottom_row[self.entering.index(v)]    

        return solution

    # Helper function for generating a square identity matrix.
    def _generate_identity(self, n):
        I = []
        for i in range(0, n):
            row = [1 if i == j else 0 for j in range(0, n)]
            I.append(row)
        return I

def run_test():
    SimplexSolver().run_simplex([[2,1], [1,2]], [4,3], [1,1])
    SimplexSolver().run_simplex(
        [[3,4,1,0,0], [3,1,0,1,0], [1,0,0,0,1]], 
        [32,17,5],
        [2,1,0,0,0]
    )
    SimplexSolver().run_simplex(
        [[1,3,3,1], [2,0,3,-1]], 
        [3,4],
        [-1,5,1,-1],
        'min'
    )

if __name__ == '__main__':
    A = []
    b = []
    c = []
    p = ''
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv,"thA:b:c:p:",["A=","b=","c=","p="])
    except getopt.GetoptError:
        print('simplex.py -A <matrix> -b <vector> -c <vector> -p <type>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('simplex.py -A <matrix> -b <vector> -c <vector> -p <obj_func_type> -t <tests>')
            sys.exit()
        elif opt == '-t':
            print('Tests... \n')
            run_test()
        elif opt in ("-A"):
            A = ast.literal_eval(arg)
        elif opt in ("-b"):
            b = ast.literal_eval(arg)
        elif opt in ("-c"):
            c = ast.literal_eval(arg)
        elif opt in ("-p"):
            p = arg.strip()
    if not A or not b or not c:
        print('Must provide arguments for A, b, c (use -h for more info) (use -t for tests)')
        sys.exit()

    # Assume maximization problem as default.
    if p not in ('max', 'min'):
        p = 'max'
    
    SimplexSolver().run_simplex(A, b, c, p)