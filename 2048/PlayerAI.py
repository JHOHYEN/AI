from random import randint
from sys import maxsize
from BaseAI import BaseAI
import time
import math

DEPTH = 4

class Node: #Use the class to manage searches
    def __init__(self,move=None,grid=None,depth=None):
        self.move = move
        self.grid = grid
        self.depth = depth

class PlayerAI(BaseAI):
    def __init__(self):
        self.movescount = 0 #movecount
        self.leavesnum = 0 #total leave
        self.depth = 4  #search depth Limit
        self.maxdepth = 0  #search depth

    def getMove(self, grid):
        maxvalue = maxsize*-1
        alpha = maxsize*-1
        beta = maxsize
        self.maxdepth = 0
        self.movescount+=1
        for move in grid.getAvailableMoves():
            movinggrid = grid.clone()
            movinggrid.move(move)
            value = self.beta(Node(move=move,grid=movinggrid,depth=DEPTH-1),alpha,beta)
            #alpha = max(value,alpha)
            if (value > alpha):
                alpha = value
            if value > maxvalue:
                maxvalue = value
                opmove = move
            if alpha >= beta:
                return opmove
            if (grid.getAvailableMoves()==None):
                print("No movements are available")
        return opmove

    def beta(self,node,alpha,beta): #betavalue(minimum value)
        if node.depth > self.maxdepth: #checking if the node is a leaf node
            self.maxdepth = node.depth #if search depth Limit > search depth then search depth = search depth Limit
        if (node.depth <= 0):
            return checkoptimum(node.grid)
        childnode = minnode(node.move,node.grid,node.depth)
        if len(childnode) == 0:  # childnode is a leaf then stop
            self.leavesnum+=1
            return checkoptimum(node.grid)
        minvalue = maxsize  #if the leaf node has not been processed
        for child in childnode:
            compare = self.alpha(child,alpha,beta)
            minvalue = min(minvalue,compare)
            if minvalue <= alpha:
                return minvalue
            beta = min(minvalue,beta)
        return minvalue

    def alpha(self,node,alpha,beta): #alphavalue(maximum value)
        self.maxdepth = min(self.maxdepth,node.depth)
        if (node.depth <= 0):
            return checkoptimum(node.grid)
        childnode = maxnode(node.move,node.grid,node.depth)
        if len(childnode) == 0:  # if childnode is a leaf then stop
            self.leavesnum+=1
            return checkoptimum(node.grid)
        maxvalue = maxsize*-1
        for child in childnode:
            comparea = self.beta(child,alpha,beta)
            maxvalue = max(maxvalue,comparea)
            alpha = max(maxvalue,alpha)
            if alpha >= beta:
                return maxvalue
        return maxvalue

def maxnode(move,grid,depth):  #get maximum childnode
    maxest = []
    for move in grid.getAvailableMoves():
        grid = grid.clone()
        grid.move(move)
        maxest.append(Node(move=move,grid=grid,depth=depth-1))
    return maxest

def minnode(move,grid,depth): #get minimum childnode
    minest = []
    for cell in grid.getAvailableCells():
        grid = grid.clone()
        grid.setCellValue(cell,2)
        minest.append(Node(move=None,grid=grid,depth=depth-1))
        grid.setCellValue(cell,4)
        minest.append(Node(move=None,grid=grid,depth=depth-1))
    return minest

def emptytiles(grid): #Number of zerotile
	freet = grid.getAvailableCells()
	if len(freet) > 0:
		return len(freet)
	else:
		return 0

def corner(grid): #Calculate whether large values ​​are on the edge
    tilem = grid.map
    len = grid.size
    score = 0
    if (tilem[0][0] == grid.getMaxTile() or tilem[0][len-1] == grid.getMaxTile() or tilem[len-1][0] == grid.getMaxTile() or tilem[len-1][len-1] == grid.getMaxTile()):
        score += 20 #if maxTile is corner then score +20
    else:
        score = grid.getMaxTile() * -10 #if maxTile is not corner then score -10
    return score

def monoto(grid): # monotonicity of board weight가 높은 쪽(구석)으로 이동을 유도
    len = grid.size
    monoscore = 0
    wgrid = [[4096,1024,256,64],[1024,256,64,16],[256,64,16,4],[64,16,4,1]]
    for row in range(len-1):
        for column in range(len-1):
            monoscore += grid.map[row][column] * wgrid[row][column]
    return monoscore

def side(grid):  #side alignment로 left side, up side 정렬 유도
    len = grid.size
    sidescore = 0
    wngrid = [[8,8,8,8],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
    wwgrid = [[8,1,1,1],[8,1,1,1],[8,1,1,1],[8,1,1,1]]
    for row in range(len-1):
        for column in range(len-1):
            sidescore += grid.map[row][column] * wngrid[row][column]
    for row in range(len-1):
        for column in range(len-1):
            sidescore += grid.map[row][column] * wwgrid[row][column]
    return sidescore

def checkoptimum(grid):  #Evaluate tile placement to induce optimal movement
    opgrade = 0
    zerotile = emptytiles(grid)
    monoscore = monoto(grid)
    cornerscore = corner(grid)
    if (grid.getMaxTile()<512):
        sidescore = side(grid)
    else:
        sidescore = 0
    #calculate final optimal score
    opgrade = (zerotile*3 + monoscore + cornerscore + sidescore)
    return opgrade
