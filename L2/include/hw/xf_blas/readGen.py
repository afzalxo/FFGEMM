
loadPatternA_lhs0 = [(0, 0), (1, 0), (0, 0), (1, 1),
                     (0, 0), (1, 0), (0, 1)]
loadPatternA_lhs1 = [(1, 1), (1, 1), (None), (None), (0, 1), (0, 0), (1, 1)]

loadPatternB_rhs0 = [(0, 0), (0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (1, 0)]
loadPatternB_rhs1 = [(1, 1), (None), (1, 1), (0, 0), (None), (0, 1), (1, 1)]

add_or_subs_b = [1, None, 0, 0, None, 1, 1]
add_or_subs_a = [1, 1, None, None, 1, 0, 0]

var_lhs = 's_lhs'
var_rhs = 's_rhs'

out_txtB = 't_subMatOps.ReadAddSubBlock(l_bAddr, (l_aColBlock << 1) + {}, (l_bColBlock << 1) + {}, (l_aColBlock << 1) + {}, (l_bColBlock << 1) + {}, l_bWordLd, {}, {});\n'
out_txtA = 't_subMatOps.ReadAddSubBlock(l_aAddr, (l_aRowBlock << 1) + {}, (l_aColBlock << 1) + {}, (l_aRowBlock << 1) + {}, (l_aColBlock << 1) + {}, l_aWordLd, {}, {});\n'

read_txtB = 't_subMatOps.ReadBlock(l_bAddr, (l_aColBlock << 1) + {}, (l_bColBlock << 1) + {}, l_bWordLd, {});\n'
read_txtA = 't_subMatOps.ReadBlock(l_aAddr, (l_aRowBlock << 1) + {}, (l_aColBlock << 1) + {}, l_aWordLd, {});\n'

for i in range(len(loadPatternA_lhs0)):
    if i not in [1, 4]:
        print(out_txtB.format(loadPatternB_rhs0[i][0], loadPatternB_rhs0[i][1], loadPatternB_rhs1[i][0], loadPatternB_rhs1[i][1], add_or_subs_b[i], var_rhs))
    else:
        print(read_txtB.format(loadPatternB_rhs0[i][0], loadPatternB_rhs0[i][1], var_rhs))

    if i not in [2, 3]:
        print(out_txtA.format(loadPatternA_lhs0[i][0], loadPatternA_lhs0[i][1], loadPatternA_lhs1[i][0], loadPatternA_lhs1[i][1], add_or_subs_a[i], var_lhs))
    else:
        print(read_txtA.format(loadPatternA_lhs0[i][0], loadPatternA_lhs0[i][1], var_lhs))
