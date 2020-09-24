# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
from util import nearestPoint
from util import Queue

DANGER_DISTANCE = 4
BFS_DEPTH = 25
OUTPUT_GATE_RANGE = 5


def createTeam(firstIndex, secondIndex, isRed, first='OffensiveAgent', second='DefensiveAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class DummyAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.team_border = []
        if self.red:
            step = 1
        else:
            step = -1
        half_width = len(gameState.data.layout.layoutText[0]) / 2
        if self.red:
            col = half_width - 1
        else:
            col = half_width
        walls = gameState.getWalls()
        for i in range(len(gameState.data.layout.layoutText)):
            if not walls[col][i] and not walls[col + step][i]:
                self.team_border.append((col, i))
        with open('log.txt', 'w')as file:
            file.write("start\n")

    def save_to_file(self, lines):
        with open('log.txt', 'a') as file:
            for line in lines:
                file.write(line + '\n')
            file.write("********************************\n")

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())
        bestAction = None
        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
        else:
            bestAction = random.choice(bestActions)
        return bestAction

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 1.0}


class OffensiveAgent(DummyAgent):
    def __init__(self, index):
        DummyAgent.__init__(self, index)
        self.actions_on_impasse = []
        self.action_resulting_impasse = None
        self.after_impasse = False
        self.lines = []
        self.prev_dis_to_ghost = 0
        self.was_pacman_fleeing = False
        self.proper_output_gate = None
        self.actions_to_be_made = []
        self.prev_pos = None
        self.action_taken = False

    def chooseAction(self, gameState):
        self.lines *= 0
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        best_action = random.choice(bestActions)
        return best_action

    def getFeatures(self, game_state, action):
        features = util.Counter()
        successor = self.getSuccessor(game_state, action)
        successor_position = successor.getAgentPosition(self.index)
        successor_food_list = self.getFood(successor).asList()
        capsules = self.getCapsules(game_state)
        cur_food_list = self.getFood(game_state).asList()
        cur_position = game_state.getAgentPosition(self.index)
        successor_enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        successor_ghosts = [a for a in successor_enemies if not a.isPacman and a.getPosition() is not None]
        distance_to_ghost = None
        cur_enemies = [game_state.getAgentState(i) for i in self.getOpponents(game_state)]
        cur_ghosts = [a for a in cur_enemies if not a.isPacman and a.getPosition() is not None]

        if self.was_pacman_fleeing and self.is_agent_in_base(cur_position):
            # Auxiliary Feature
            if self.proper_output_gate is None:
                self.proper_output_gate = self.get_gate_in_range(cur_position, OUTPUT_GATE_RANGE)
                self.actions_to_be_made = self.get_actions_to_destination(self.proper_output_gate, game_state)
            if self.prev_pos != cur_position:
                self.action_taken = False
                self.prev_pos = cur_position
            if not self.action_taken:
                move = self.actions_to_be_made[0]
                if move == action:
                    features['to_proper_output_gate'] = 0
                    self.actions_to_be_made.pop(0)
                    self.action_taken = True
                    if successor_position == self.proper_output_gate:
                        self.proper_output_gate = None
                        self.actions_to_be_made = []
                        self.was_pacman_fleeing = False
                else:
                    features['to_proper_output_gate'] = 1
            else:
                features['to_proper_output_gate'] = 1
        else:
            is_ghost_chasing_pacman = False
            cur_dis_to_ghost = [self.getMazeDistance(cur_position, a.getPosition()) for a in cur_ghosts]
            if len(cur_ghosts) > 0:
                if min(cur_dis_to_ghost) <= 3 and not self.is_agent_in_base(cur_position):
                    is_ghost_chasing_pacman = True

            # Feature 1 - successorScore
            features['successor_score'] = len(successor_food_list)

            if not is_ghost_chasing_pacman:
                # Feature 2 - closest_food
                if len(successor_food_list) > 0:
                    min_distance_food = min([self.getMazeDistance(successor_position, food) for food in cur_food_list])
                    features["closest_food"] = min_distance_food

            # Feature 3 - closest_capsule
            if len(capsules) > 0:
                min_distance_capsule = min([self.getMazeDistance(successor_position, capsule) for capsule in capsules])
                features['closest_capsule'] = min_distance_capsule

            # Feature 4 - return_to_base
            if successor.getAgentState(self.index).isPacman:
                num_carrying = successor.getAgentState(self.index).numCarrying
                if num_carrying:
                    min_dis_to_base = min(
                        [self.getMazeDistance(successor_position, base_gate) for base_gate in self.team_border])
                    features['return_to_base'] = pow(min_dis_to_base, 0.5) * num_carrying

            # Feature 5 - immediate_return_to_base
            if len(capsules) == 0 and is_ghost_chasing_pacman:
                min_dis_to_base = min(
                    [self.getMazeDistance(successor_position, base_gate) for base_gate in self.team_border])
                features['immediate_return_to_base'] = min_dis_to_base

            # Feature 6 - distance_to_ghost
            if len(cur_ghosts) > 0:
                distance_to_ghost = min(
                    [self.getMazeDistance(successor_position, a.getPosition()) for a in cur_ghosts])
                if distance_to_ghost <= DANGER_DISTANCE:
                    features['distance_to_ghost'] = distance_to_ghost
            elif self.prev_dis_to_ghost != 0:
                features['distance_to_ghost'] = self.prev_dis_to_ghost
                self.prev_dis_to_ghost = 0

            if len(cur_ghosts) > 0:
                if cur_ghosts[0].scaredTimer > 1.5 * distance_to_ghost:
                    if self.prev_dis_to_ghost == 0:
                        features['distance_to_ghost'] = distance_to_ghost    # check it out
                    else:
                        features['distance_to_ghost'] = - distance_to_ghost

            # Feature 7 - is_impasse
            foods_capsules = cur_food_list[:]
            foods_capsules.extend(capsules)
            states_exist_in_depth, is_there_food = self.get_impasse(successor, cur_position, BFS_DEPTH, foods_capsules)
            if not is_there_food and states_exist_in_depth:
                features["is_impasse"] = 1

            # Feature 8 - dead_impasse
            if len(successor_ghosts) > 0:
                if states_exist_in_depth:
                    if distance_to_ghost <= 2 * states_exist_in_depth + 1:
                        features['dead_impasse'] = 1

            if distance_to_ghost is not None:
                self.prev_dis_to_ghost = distance_to_ghost

            self.was_pacman_fleeing = is_ghost_chasing_pacman
        self.prev_pos = cur_position
        return features

    def getWeights(self, gameState, action):
        return {'successor_score': -100, 'closest_food': -2, 'dead_impasse': -1000, 'distance_to_ghost': 100,
                'is_impasse': -10000, 'closest_capsule': -30, 'return_to_base': -3, 'to_proper_output_gate': -5}

    def get_impasse(self, successor, current_position, depth, foods_capsules):
        is_there_food = False
        bfs_queue = Queue()
        bfs_queue.push(successor)
        already_seen = set()
        already_seen.add(current_position)
        for i in range(1, depth):
            if not bfs_queue.isEmpty():
                successor = bfs_queue.pop()
                successor_pos = successor.getAgentPosition(self.index)
                if successor_pos in foods_capsules:
                    is_there_food = True
                already_seen.add(successor_pos)
                legal_actions = successor.getLegalActions(self.index)
            else:
                break
            for action in legal_actions:
                if action != Directions.STOP:
                    temp_successor = self.getSuccessor(successor, action)
                    successor_pos = temp_successor.getAgentPosition(self.index)
                    if successor_pos not in already_seen:
                        bfs_queue.push(temp_successor)
        if bfs_queue.isEmpty():
            return len(already_seen), is_there_food
        else:
            return False, is_there_food

    def is_impasse(self, successor):
        legal_actions = successor.getLegalActions(self.index)
        if len(legal_actions) == 2:
            return True
        return False

    def is_agent_in_base(self, cur_position):
        if self.red:
            return cur_position[0] <= self.team_border[0][0]
        else:
            return cur_position[0] >= self.team_border[0][0]

    def get_gate_in_range(self, cur_position, RANGE):
        while RANGE > 0:
            for gate in self.team_border:
                if self.getMazeDistance(cur_position, gate) >= RANGE:
                    return gate
            RANGE -= 1
        return cur_position

    def get_actions_to_destination(self, proper_output_gate, cur_state):
        queue = Queue()
        queue.push(cur_state)
        visited = set()
        child_to_parent = {cur_state: (None, None)}
        while not queue.isEmpty():
            node = queue.pop()
            node_pos = node.getAgentPosition(self.index)
            if node_pos == proper_output_gate:
                actions = []
                while child_to_parent[node][0] is not None:
                    (node, action) = child_to_parent[node]
                    actions.append(action)
                actions.reverse()
                return actions
            visited.add(node_pos)
            legal_actions = node.getLegalActions(self.index)
            for action in legal_actions:
                if action != Directions.STOP:
                    temp_successor = self.getSuccessor(node, action)
                    successor_pos = temp_successor.getAgentPosition(self.index)
                    if successor_pos not in visited and self.is_agent_in_base(successor_pos):
                        queue.push(temp_successor)
                        child_to_parent[temp_successor] = (node, action)


class DefensiveAgent(DummyAgent):
    def __init__(self, index):
        DummyAgent.__init__(self, index)
        self.prev_eaten_food = None
        self.prev_food_list = []
        self.prev_opp_agents_mode = {}

    """
       My defensive algorithm.
    """

    def getFeatures(self, game_state, action):
        self.get_border_foods(game_state)
        features = util.Counter()
        cur_pos = game_state.getAgentPosition(self.index)
        cur_capsules = self.getCapsulesYouAreDefending(game_state)
        successor = self.getSuccessor(game_state, action)
        successor_pos = successor.getAgentPosition(self.index)
        successor_defending_food_list = self.getFoodYouAreDefending(successor).asList()
        successor_defending_capsules_list = self.getCapsulesYouAreDefending(successor)
        successor_state = successor.getAgentState(self.index)
        is_scared = successor_state.scaredTimer
        eaten_foods = list(set(self.prev_food_list) - set(successor_defending_food_list))
        min_dist_to_pacman, invaders = self.get_min_distance_to_pacman(game_state, successor_pos)

        if self.prev_eaten_food == cur_pos:
            self.prev_eaten_food = None

        # Feature 1 - onDefense
        features['on_defense'] = 1
        if successor_state.isPacman: features['on_defense'] = 0

        # Feature 2 - numInvaders
        if len(invaders) > 0:
            features['num_invaders'] = len(invaders)
            features['invader_distance'] = min_dist_to_pacman

        # Feature 3 & 4 - invaderDistance & is_impasse
        if is_scared and min_dist_to_pacman > 3:
            features['invader_distance'] *= -1
            features['num_invaders'] = 0
            is_impasse = self.get_impasse(successor, cur_pos, BFS_DEPTH)
            if is_impasse:
                features['is_impasse'] = 1

        # Feature 5 - dis_to_eaten_food
        if 'num_invaders' not in features:
            if len(eaten_foods) > 0:
                eaten_food = eaten_foods[0]
                self.prev_eaten_food = eaten_food
                features['dis_to_eaten_food'] = self.getMazeDistance(eaten_food, successor_pos)
            elif self.prev_eaten_food is not None:
                features['dis_to_eaten_food'] = self.getMazeDistance(self.prev_eaten_food, successor_pos)
            else:
                border_foods = self.get_border_foods(game_state)
                flag = False
                if len(border_foods):
                    border_foods_col = border_foods[0][0]
                    if len(cur_capsules) > 0:
                        if self.red and max([i[0] for i in cur_capsules]) >= border_foods_col or\
                                        not self.red and min([i[0] for i in cur_capsules]) <= border_foods_col:
                            features['dist_to_defending_capsule'] = min([self.getMazeDistance(successor_pos, pos) for pos in
                                                                 cur_capsules])
                            flag = True
                    if not flag:
                        features['distance_to_border_food'] = min([self.getMazeDistance(successor_pos, border_food) for
                                                                   border_food in border_foods])
        self.prev_food_list = successor_defending_food_list
        return features

    def get_min_distance_to_pacman(self, state, successor_pos):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]
        if len(invaders) > 0:
            return min([self.getMazeDistance(successor_pos, a.getPosition()) for a in invaders]), invaders
        else:
            return None, []

    def get_impasse(self, successor, current_position, depth):
        bfs_queue = Queue()
        bfs_queue.push(successor)
        already_seen = set()
        already_seen.add(current_position)
        for i in range(1, depth):
            if not bfs_queue.isEmpty():
                successor = bfs_queue.pop()
                successor_pos = successor.getAgentPosition(self.index)
                already_seen.add(successor_pos)
                legal_actions = successor.getLegalActions(self.index)
            else:
                break
            for action in legal_actions:
                if action != Directions.STOP:
                    temp_successor = self.getSuccessor(successor, action)
                    successor_pos = temp_successor.getAgentPosition(self.index)
                    if successor_pos not in already_seen:
                        bfs_queue.push(temp_successor)
        if bfs_queue.isEmpty():
            return True
        else:
            return False

    def getWeights(self, gameState, action):
        return {'num_invaders': -10000, 'on_defense': 10000,
                'invader_distance': -100, 'is_impasse': -10000,
                'dis_to_eaten_food': -10, 'dist_to_defending_capsule': -30,
                'distance_to_border_food': -20}

    def get_border_foods(self, game_state):
        team_border_col = self.team_border[0][0]
        border_foods = []
        foods = game_state.data.food
        if self.red:
            col_range = range(team_border_col, -1, -1)
        else:
            col_range = range(team_border_col, foods.width)
        for i in col_range:
            for j in range(len(foods.data[i])):
                if foods.data[i][j]:
                    border_foods.append((i, j))
            if len(border_foods) != 0:
                return border_foods
        return []


