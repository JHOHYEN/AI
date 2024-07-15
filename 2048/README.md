# 2048

This is the code to write PlayerAI.py, which intelligently plays the 2048 puzzle game. Below is a startup code snippet that allows you to observe what the game looks like when it runs. Here's the "naive" player AI: The getMove() function randomly selects the next move from among the available moves.

```
from random import randint 
from BaseAI import BaseAI 
class PlayerAI(BaseAI): 
    def getMove(self, grid): 
        moves = grid.getAvailableMoves() 
        return moves[randint(0, len(moves) - 1)] if moves else None 
```

Of course, that's actually a very naive way to play the 2048 puzzle game. Player AI should be implemented with the following in mind:

* Employ the minimax algorithm. This is a requirement. There are many viable strategies to 
beat the 2048-puzzle game, but in this assignment we will be practicing with the minimax 
algorithm. 

* Implement alpha-beta pruning. This is a requirement. This should speed up the search 
process by eliminating irrelevant branches. In this case, is there anything we can do about 
move ordering? 

* Depth and random seed of the algorithm. The depth of minimax algorithm is 100, and the 
seed for randomizing the tile is 218. You MUST NOT change this parameter, otherwise 
you will get zero.

* Use heuristic functions. To be able to cut off your search at any point, you must employ 
heuristic functions to allow you to assign approximate values to nodes in the tree. Remember, 
the time limit allowed for each move is 1 seconds, so you must implement a systematic way 
to cut off your search before time runs out.

* Assign heuristic weights. You will likely want to include more than one heuristic function. In 
that case, you will need to assign weights associated with each individual heuristic. Deciding 
on  an  appropriate  set  of  weights  will  take  careful  reasoning,  along  with  careful 
experimentation. If you feel adventurous, you can also simply write an optimization meta- 
algorithm to iterate over the space of weight vectors, until you arrive at results that you are 
happy enough with.