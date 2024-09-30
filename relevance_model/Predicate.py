
RELATION_ATTRIBUTE = '___'

class Predicate(object):
    def __init__(self, predicate_str):
        self.index1 = None
        self.index2 = None
        self.operator = None
        self.operand1 = None
        self.operand2 = None
        self.constant = None
        res = self.parsePredicate(predicate_str)
        print("Parse predicate : {}, {}".format(predicate_str, res))
        self.index1 = res[0]
        self.operand1 = {'relation': res[1].split(RELATION_ATTRIBUTE)[0].strip(),
                         'attribute': res[1].split(RELATION_ATTRIBUTE)[1].strip()}
        self.operator = res[2]
        self.index2 = res[3]
        if self.index1 == self.index2:
            # constant
            self.constant = res[4]
        else:
            # operand2
            self.operand2 = {'relation': res[4].split(RELATION_ATTRIBUTE)[0].strip(),
                             'attribute': res[4].split(RELATION_ATTRIBUTE)[1].strip()}

    def print(self):
        # constant predicates
        if self.constant != None:
            return self.index1 + '.' + self.operand1['relation'] + '.' + self.operand1[
                'attribute'] + ' ' + self.operator + ' ' + self.constant
        else:
            return self.index1 + '.' + self.operand1['relation'] + '.' + self.operand1[
                'attribute'] + ' ' + self.operator + ' ' + self.index2 + '.' + self.operand2['relation'] + '.' + \
                self.operand2['attribute']

    def parsePredicate(self, predicate):
        """
        :param predicate:
        :return:
        """
        if predicate.find("(") != -1 and predicate.find(")") != -1 and predicate[:2] != 'ML' and predicate[:len(
                'similar')] != 'similar':
            ss = predicate[predicate.find("(") + 1:predicate.find(")")]
            for i in range(1, len(ss), 1):
                if ss[:i].isalpha() and ss[i:].isdigit():
                    print(ss, i, ss[:i], ss[i:], ss[:i].isalpha, ss[i:].isdigit)
                    return None

        res = None
        # check ML(t0.A, t1.B)
        if predicate[:2] == "ML":
            t = predicate.split('(')
            operator = t[0]
            # find substring in '( ... )'
            op = predicate[predicate.find("(") + 1:predicate.find(")")]
            operand1, operand2 = op.split(',')[0].strip(), op.split(',')[1].strip()
            res = [operand1, operator, operand2]
        elif len(predicate) >= len('similar') and predicate[:len('similar')] == "similar":
            ss = [e.strip() for e in predicate[len('similar') + 1:-1].split(',')]
            # print(ss)
            op = ss[0] + ' ' + ss[-1]
            res = [ss[1], op, ss[2]]
        else:  # check other predicates
            t = predicate.split()
            operand1 = t[0]
            operator = t[1]
            operand2 = ' '.join(t[2:])
            res = [operand1, operator, operand2]

        # process relation operand1 and operand2 and constant
        # relation.tx.attribute or constant (assume that constant does not have ".")
        operand1_ = res[0].split(".")
        index1_ = operand1_[1]
        operand1_new = operand1_[0] + RELATION_ATTRIBUTE + operand1_[2]
        operator_ = res[1]
        operand2_ = res[2].split(".")
        if len(operand2_) != 3 or (len(operand2_) > 1 and operand2_[1][:1] != 't'):
            index2_ = index1_
            operand2_new = res[2]
        else:
            index2_ = operand2_[1]
            operand2_new = operand2_[0] + RELATION_ATTRIBUTE + operand2_[2]

        return [index1_, operand1_new, operator_, index2_, operand2_new]

    def isValidRelation(self, t_index, relation):
        if self.index1 == t_index and self.operand1['relation'] == relation:
            return True
        elif self.index2 == t_index and self.operand2 != None and self.operand2['relation'] == relation:
            return True
        else:
            return False

    def getRelations(self):
        if self.index1 != self.index2:
            return {self.index1: self.operand1['relation'], self.index2: self.operand2['relation']}
        else:
            return {self.index1: self.operand1['relation']}

    def isConstantPredicate(self):
        if self.constant == None:
            return False
        return True
