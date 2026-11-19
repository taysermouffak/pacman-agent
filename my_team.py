# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
from collections import deque

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    """
    Returns the two agents of the team: one attacker and one defender.
    The tournament calls this function without extra arguments, so the
    default values are the team that will play.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class BaseAgent(CaptureAgent):
    """
    Common tools for both agents.

    Instead of the reflex agent of the template (one-step look-ahead with
    weighted features), our agents decide a set of target cells and then
    move along the shortest path to them, computed with breadth-first search
    (the same graph search as in Lab 1). Cells that are dangerous can be
    removed from the search, so the path avoids them.
    """

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()
        width = self.walls.width

        # The red team owns the left half of the map and the blue team the
        # right half. border_x is the last column of our own half.
        self.border_x = width // 2 - 1 if self.red else width // 2
        self.home_border = [(self.border_x, y) for y in range(self.walls.height)
                            if not self.walls[self.border_x][y]]

    # ---------- Map helpers ----------

    def is_home(self, pos):
        """True if the cell belongs to our half of the map."""
        return pos[0] <= self.border_x if self.red else pos[0] >= self.border_x

    def neighbours(self, pos):
        """Cells reachable in one move from pos, with the action needed."""
        x, y = pos
        result = []
        for action in (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST):
            dx, dy = Actions.direction_to_vector(action)
            nx, ny = int(x + dx), int(y + dy)
            if not self.walls[nx][ny]:
                result.append(((nx, ny), action))
        return result

    def my_position(self, game_state):
        return nearest_point(game_state.get_agent_state(self.index).get_position())

    def distance_home(self, pos):
        return min(self.get_maze_distance(pos, b) for b in self.home_border)

    # ---------- Path planning ----------

    def first_step_towards(self, start, targets, blocked=frozenset()):
        """
        Breadth-first graph search from start to the closest cell in targets,
        never entering cells in blocked. Returns the first action of that
        shortest path, or None if no target can be reached.
        """
        targets = set(targets)
        if not targets:
            return None
        frontier = deque()
        visited = {start}
        for cell, action in self.neighbours(start):
            if cell not in blocked:
                visited.add(cell)
                frontier.append((cell, action))
        while frontier:
            cell, first_action = frontier.popleft()
            if cell in targets:
                return first_action
            for nxt, _ in self.neighbours(cell):
                if nxt not in visited and nxt not in blocked:
                    visited.add(nxt)
                    frontier.append((nxt, first_action))
        return None

    # ---------- Opponent helpers ----------

    def visible_enemies(self, game_state):
        """(index, state, position) of the opponents we can currently see."""
        result = []
        for i in self.get_opponents(game_state):
            state = game_state.get_agent_state(i)
            pos = state.get_position()
            if pos is not None:
                result.append((i, state, nearest_point(pos)))
        return result

    def dangerous_ghosts(self, game_state):
        """Positions of visible enemy ghosts that can eat us (not scared)."""
        return [pos for _, state, pos in self.visible_enemies(game_state)
                if not state.is_pacman and state.scared_timer <= 2]

    def visible_invaders(self, game_state):
        """Positions of visible enemy Pacmen inside our half."""
        return [pos for _, state, pos in self.visible_enemies(game_state) if state.is_pacman]

    def safe_fallback(self, game_state, threats):
        """
        Used when no path is safe: pick the legal move that keeps the largest
        distance to the threats, breaking ties by getting closer to home.
        """
        actions = [a for a in game_state.get_legal_actions(self.index) if a != Directions.STOP]
        if not actions:
            return Directions.STOP
        my_pos = self.my_position(game_state)
        best, best_score = None, None
        for action in actions:
            dx, dy = Actions.direction_to_vector(action)
            pos = (int(my_pos[0] + dx), int(my_pos[1] + dy))
            danger = min((self.get_maze_distance(pos, t) for t in threats), default=99)
            score = (danger, -self.distance_home(pos))
            if best_score is None or score > best_score:
                best, best_score = action, score
        return best


class OffensiveAgent(BaseAgent):
    """
    Attacker: eats enemy food, avoids enemy ghosts and comes back home to
    deposit the food before it is too risky (food is only scored when it is
    brought back to our side).
    """

    CARRY_LIMIT = 6  # food carried before returning home when not in danger

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = self.my_position(game_state)
        carrying = my_state.num_carrying
        food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        ghosts = self.dangerous_ghosts(game_state)
        ghost_dist = min((self.get_maze_distance(my_pos, g) for g in ghosts), default=99)

        # While we are a ghost on our own side, eat an invader that is close.
        invaders = self.visible_invaders(game_state)
        if not my_state.is_pacman and my_state.scared_timer == 0 and invaders:
            close = [p for p in invaders if self.get_maze_distance(my_pos, p) <= 4]
            if close:
                action = self.first_step_towards(my_pos, close, self.enemy_side_cells())
                if action is not None:
                    return action

        # Cells next to a dangerous ghost are removed from the search, but only
        # on the enemy side: on our side we are a ghost and cannot be eaten.
        blocked = set()
        for g in ghosts:
            for cell in [g] + [c for c, _ in self.neighbours(g)]:
                if not self.is_home(cell):
                    blocked.add(cell)

        # Decide whether to go back home. Each agent moves once every 4 turns,
        # so timeleft // 4 is the number of moves we still have.
        moves_left = game_state.data.timeleft // 4
        dist_home = self.distance_home(my_pos)
        go_home = carrying > 0 and (
            len(food) <= 2
            or carrying >= self.CARRY_LIMIT
            or ghost_dist <= 5
            or dist_home + 5 >= moves_left)

        if go_home:
            targets = self.home_border
        elif ghost_dist <= 5 and capsules:
            # Chased with a capsule available: eating it scares the ghosts.
            targets = capsules
        else:
            targets = food

        action = self.first_step_towards(my_pos, targets, blocked)
        if action is None and targets is not self.home_border:
            # The food is not safely reachable: retreat home instead.
            action = self.first_step_towards(my_pos, self.home_border, blocked)
        if action is None:
            action = self.safe_fallback(game_state, ghosts)
        return action

    def enemy_side_cells(self):
        """All free cells of the enemy half (used to keep a ghost at home)."""
        return {(x, y) for x in range(self.walls.width) for y in range(self.walls.height)
                if not self.walls[x][y] and not self.is_home((x, y))}


class DefensiveAgent(BaseAgent):
    """
    Defender: never leaves our half. It chases visible invaders, goes to
    the place where our food has just been eaten (a sign of an invisible
    invader) and otherwise patrols the border near our food.
    """

    def register_initial_state(self, game_state):
        BaseAgent.register_initial_state(self, game_state)
        self.enemy_side = {(x, y) for x in range(self.walls.width) for y in range(self.walls.height)
                           if not self.walls[x][y] and not self.is_home((x, y))}
        self.last_eaten = None  # last cell where we lost food
        self.patrol_target = None

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = self.my_position(game_state)
        invaders = self.visible_invaders(game_state)
        self.update_last_eaten(game_state, my_pos)

        if invaders:
            self.last_eaten = None
            if my_state.scared_timer > 0:
                # Scared: an invader would eat us, so stay two steps away
                # and wait for the timer to run out.
                blocked = set(self.enemy_side)
                for p in invaders:
                    blocked.add(p)
                    blocked.update(c for c, _ in self.neighbours(p))
                targets = [c for c in self.cells_at_distance(invaders, 2) if c not in blocked]
                action = self.first_step_towards(my_pos, targets, blocked)
                return action if action is not None else self.safe_fallback(game_state, invaders)
            targets = invaders
        elif self.last_eaten is not None:
            targets = [self.last_eaten]
        else:
            targets = [self.choose_patrol_target(game_state, my_pos)]

        action = self.first_step_towards(my_pos, targets, self.enemy_side)
        if action is None:
            legal = [a for a in game_state.get_legal_actions(self.index) if a != Directions.STOP]
            action = random.choice(legal) if legal else Directions.STOP
        return action

    def update_last_eaten(self, game_state, my_pos):
        """Remembers where our food disappeared since the previous turn."""
        previous = self.get_previous_observation()
        if previous is not None:
            before = set(self.get_food_you_are_defending(previous).as_list())
            now = set(self.get_food_you_are_defending(game_state).as_list())
            eaten = before - now
            if eaten:
                self.last_eaten = min(eaten, key=lambda f: self.get_maze_distance(my_pos, f))
        if self.last_eaten == my_pos:
            self.last_eaten = None

    def choose_patrol_target(self, game_state, my_pos):
        """
        Patrols the border cell closest (on average) to the food we defend.
        When it gets there, it moves to a nearby border cell so it keeps
        moving along the frontier.
        """
        food = self.get_food_you_are_defending(game_state).as_list()
        if not food:
            return random.choice(self.home_border)
        best = min(self.home_border,
                   key=lambda b: sum(self.get_maze_distance(b, f) for f in food))
        if self.patrol_target is None or my_pos == self.patrol_target:
            nearby = [b for b in self.home_border if self.get_maze_distance(b, best) <= 4]
            self.patrol_target = random.choice(nearby) if nearby else best
        return self.patrol_target

    def cells_at_distance(self, sources, d):
        """Free cells of our half at exactly maze distance d from a source."""
        cells = []
        for x in range(self.walls.width):
            for y in range(self.walls.height):
                if not self.walls[x][y] and self.is_home((x, y)):
                    if min(self.get_maze_distance((x, y), s) for s in sources) == d:
                        cells.append((x, y))
        return cells
