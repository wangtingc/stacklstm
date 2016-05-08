stack_lstm_decoder: 
    1. given 'f' list and assume that the tree is the binarized tree, we can always reconstruct the unique parse tree
    2. given 'f' list but not constraint with binarized tree, the parse tree can not be reconstructed. there will be multiple feasible trees. E.g. given [0,1,1,1,0,1,1,0,1,1], we can construct two trees [root, [node1, ['a', 'b', 'c']], [node2, ['d', 'e']], [node3, ['f', 'g']]] and [root, [node1, ['a', 'b', 'c']], [node2, ['c', 'd'], [node3, ['f', 'g']]]].
        2.1 in this case, at each time-step, we need to choose the only state in the stack to be the father of resulting node.
