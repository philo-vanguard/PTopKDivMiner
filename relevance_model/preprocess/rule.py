class Predicate:
    def __init__(self, tableName=None):
        self.index1 = None
        self.index2 = None
        self.attr1 = ""
        self.attr2 = ""
        self.constant = ""
        self.operator = ""
        self.tableName = tableName
        self.confidence = 0.0 
        self.type = ""

    def transform(self, str_predicate, op):
        info = str_predicate.strip().split(op)
        self.index1 = int(info[0].strip().split(".")[1][1])
        self.attr1 = info[0].strip().split(".")[2]
        self.operator = op
        if "t1." in info[1]:
            self.index2 = int(info[1].strip().split(".")[1][1])
            self.attr2 = info[1].strip().split(".")[2]
            self.type = "non-constant"
        else:
            self.constant = eval(info[1].strip()) if "\'" in info[1] else info[1].strip()
            self.type = "constant"

    def get_index1(self):
        return self.index1

    def get_index2(self):
        return self.index2

    def get_attr1(self):
        return self.attr1

    def get_attr2(self):
        return self.attr2

    def get_constant(self):
        return self.constant

    def get_operator(self):
        return self.operator

    def get_type(self):
        return self.type

    def is_constant(self):
        if self.type == "non-constant":
            return False
        else:
            return True

    def get_confidence(self):
        return self.confidence

    def assign_info(self, index1, index2, attr1, attr2, constant, operator, p_type):
        self.index1 = index1
        self.index2 = index2
        self.attr1 = attr1
        self.attr2 = attr2
        self.constant = constant
        self.operator = operator
        self.type = p_type

    def print_predicate(self):
        output = ""
        if self.type == "non-constant":
            output = "t" + str(self.index1) + "." + self.attr1 + " = t" + str(self.index2) + "." + self.attr2
        elif self.type == "constant":
            output = "t" + str(self.index1) + "." + self.attr1 + " = " + str(self.constant)
        elif self.type == "Mc":
            output = "Mc(t" + str(self.index1) + ".[" + ", ".join(self.attr1) + "], t" + str(self.index1) + "." + self.attr2 + ") > " + str(self.confidence)
        return output

    def print_predicate_new(self):
        output = ""
        if self.type == "non-constant":
            output = self.tableName + ".t" + str(self.index1) + "." + self.attr1 + " == " + self.tableName + ".t" + str(self.index2) + "." + self.attr2
        elif self.type == "constant":
            output = self.tableName + ".t" + str(self.index1) + "." + self.attr1 + " == " + str(self.constant)
        return output


class REELogic:
    def __init__(self):
        self.type = "logic"
        self.currents = []
        self.RHS = None
        self.support = None
        self.confidence = None
        self.tuple_variable_cnt = 0

    # identify precondition X and consequence e based on textual rule
    def load_X_and_e(self, textual_rule):
        precondition, rhs_info = textual_rule.split("->")[0].strip().split(":")[1].strip(), textual_rule.split("->")[1].strip().split(",")
        consequence = rhs_info[0].strip()

        if '⋀' in precondition:
            precondition = precondition.split('⋀')
        elif '^' in precondition:
            precondition = precondition.split('^')
        else:
            precondition = [precondition]
        for idx in range(len(precondition)):
            predicate = precondition[idx].strip()
            operator = self.obtain_operator(predicate)
            p = []
            if operator != '':  # if operator is one of <>, >=, <=, ==, =, > and <
                pre = Predicate()
                pre.transform(predicate, operator)
                self.currents.append(pre)

        # obtain consequence e
        operator = self.obtain_operator(consequence)
        self.RHS = Predicate()
        self.RHS.transform(consequence, operator)
        self.support = float(rhs_info[1].split(':')[1].strip())  # the suport of this REE
        self.confidence = float(rhs_info[2].split(':')[1].strip())  # the confidence of this REE


    # identify the operator from <>, >=, <=, =, > and <
    def obtain_operator(self, predicate):
        operator = ''
        if (predicate.find('<>') != -1):
            operator = '<>'
        elif (predicate.find('>=') != -1):
            operator = '>='
        elif (predicate.find('<=') != -1):
            operator = '<='
        elif (predicate.find('==') != -1):
            operator = '=='
        elif (predicate.find('=') != -1):
            operator = '='
        elif (predicate.find('>') != -1):
            operator = '>'
        elif (predicate.find('<') != -1):
            operator = '<'
        return operator

    def get_support(self):
        return self.support

    def get_confidence(self):
        return self.confidence

    def get_type(self):
        return self.type

    def get_tuple_variable_cnt(self):
        return self.tuple_variable_cnt

    def is_constant(self):
        if self.tuple_variable_cnt == 0:
            self.tuple_variable_cnt = 1
            for pred in self.currents:
                if not pred.is_constant():
                    self.tuple_variable_cnt = 2
            if not self.RHS.is_constant():
                self.tuple_variable_cnt = 2

        return self.tuple_variable_cnt

    def get_currents(self):
        return self.currents

    def get_RHS(self):
        return self.RHS

    def print_rule(self):
        output = ""

        output += self.currents[0].print_predicate()
        for idx in range(1, len(self.currents)):
            output += " ^ "
            output += self.currents[idx].print_predicate()

        output += " -> "
        output += self.RHS.print_predicate()

        return output

