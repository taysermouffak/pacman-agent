# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
from collections import deque
from functools import lru_cache

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
    # Resolve only the agent classes defined in this file. The tournament
    # uses the defaults, while the mapping still supports explicit options.
    agent_classes = {'OffensiveAgent': OffensiveAgent,
                     'DefensiveAgent': DefensiveAgent}
    return [agent_classes[first](first_index), agent_classes[second](second_index)]


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
        # right half. Keep only border cells with an open crossing.
        # border_x is the last column of our own half.
        self.border_x = width // 2 - 1 if self.red else width // 2
        self.home_border = [(self.border_x, y) for y in range(self.walls.height)
                            if not self.walls[self.border_x][y]
                            and not self.walls[self.border_x + (1 if self.red else -1)][y]]

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

    def safe_fallback(self, game_state, threats, stay_home=False):
        """
        Used when no path is safe: pick the legal move that keeps the largest
        distance to the threats, breaking ties by getting closer to home.
        """
        my_pos = self.my_position(game_state)
        actions = []
        for action in game_state.get_legal_actions(self.index):
            if action == Directions.STOP:
                continue
            dx, dy = Actions.direction_to_vector(action)
            next_pos = (int(my_pos[0] + dx), int(my_pos[1] + dy))
            if not stay_home or self.is_home(next_pos):
                actions.append(action)
        if not actions:
            return Directions.STOP
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

    # Deposit after a larger haul when the route is safe. Nearby ghosts,
    # nearly exhausted food and the game clock can still trigger an earlier return.
    CARRY_LIMIT = 11

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.recent_positions = deque(maxlen=12)
        self.detour = None

    def choose_action(self, game_state):
        action = self.plan_action(game_state)
        state = game_state.get_agent_state(self.index)
        if not state.is_pacman:
            return action
        pos = self.my_position(game_state)
        guards = [(q, enemy.scared_timer)
                  for _, enemy, q in self.visible_enemies(game_state)
                  if not enemy.is_pacman and self.get_maze_distance(pos, q) <= 7]
        if not guards:
            return action
        return self.avoid_trap(game_state, pos, action, guards)

    def avoid_trap(self, state, pos, action, guards):
        """Check whether nearby ghosts could cut off the chosen route."""
        capsules = set(self.get_capsules(state))
        horizon = 6
        legal = state.get_legal_actions(self.index)
        choices = [(point, move) for point, move in self.neighbours(pos) if move in legal]
        if not choices:
            return action
        threat_eta = {}

        def danger_at(point):
            # A ghost cannot capture us before it arrives and its timer expires.
            if point not in threat_eta:
                threat_eta[point] = min(max(self.get_maze_distance(point, ghost), timer)
                                        for ghost, timer in guards)
            return threat_eta[point]

        def continuation(start):
            queue = deque([(start, 1)])
            visited = {start}
            result = 0
            clearance = -10
            while queue:
                point, step = queue.popleft()
                # Crossing home or taking a capsule ends this threat model.
                if self.is_home(point) or point in capsules:
                    return horizon + 1, -step
                margin = danger_at(point) - step
                if margin <= 0:
                    continue
                result = max(result, step)
                clearance = max(clearance, margin)
                if step < horizon:
                    for cell, _ in self.neighbours(point):
                        if cell not in visited:
                            visited.add(cell)
                            queue.append((cell, step + 1))
            return result, clearance
        ranks = {move: continuation(point) for point, move in choices}
        planned = ranks.get(action, (0, 0))
        carrying = state.get_agent_state(self.index).num_carrying
        if planned[0] >= horizon or (planned[0] >= 3 and not carrying):
            return action
        best = max(choices, key=lambda item: (
            ranks[item[1]], item[1] == action,
            -self.distance_home(item[0]),
            -self.recent_positions.count(item[0])))
        return best[1] if ranks[best[1]][0] > planned[0] else action

    def plan_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = self.my_position(game_state)
        self.recent_positions.append(my_pos)
        carrying = my_state.num_carrying
        food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        ghosts = self.dangerous_ghosts(game_state)
        enemy_states = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        guard_timers = [enemy.scared_timer for enemy in enemy_states if not enemy.is_pacman]
        power_window = min(guard_timers, default=0)
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

        # A capsule takes effect before collision is checked.
        blocked.difference_update(capsules)

        # If a guard is holding this entrance, pick another open passage.
        # Keep the new goal until we reach it, then recheck the crossing.
        if not my_state.is_pacman and food:
            if self.detour == my_pos:
                entry = (my_pos[0] + (1 if self.red else -1), my_pos[1])
                self.detour = None
                if entry not in blocked:
                    return Directions.EAST if self.red else Directions.WEST
            if self.detour is None and len(self.recent_positions) >= 8 and (
                    len(set(self.recent_positions)) <= 4 or ghost_dist <= 2):
                def clearance(gate):
                    enemy_cell = (gate[0] + (1 if self.red else -1), gate[1])
                    danger = min((self.get_maze_distance(enemy_cell, g) for g in ghosts), default=9)
                    return min(danger, 8) * 2 - self.get_maze_distance(my_pos, gate) - (
                        8 if gate in self.recent_positions else 0)
                choices = [gate for gate in self.home_border if gate != my_pos]
                if choices:
                    self.detour = max(choices, key=clearance)
            if self.detour is not None:
                detour_action = self.first_step_towards(my_pos, [self.detour], self.enemy_side_cells())
                if detour_action is not None:
                    return detour_action
                self.detour = None
        else:
            self.detour = None

        # Decide whether to go back home. Each agent moves once every 4 turns,
        # so timeleft // 4 is the number of moves we still have.
        moves_left = game_state.data.timeleft // 4
        dist_home = self.distance_home(my_pos)
        # Make longer raids while guards are scared, or when repeated trips
        # would be expensive and no active guard is visible.
        carry_limit = 18 if power_window >= dist_home + 7 else self.CARRY_LIMIT
        if not guard_timers or (not ghosts and dist_home >= 9):
            carry_limit = max(self.CARRY_LIMIT, min(24, dist_home * 2))
        score = self.get_score(game_state)
        their_load = sum(enemy.num_carrying for enemy in enemy_states)
        # Deposit a lead before the clock runs out or another raid can overtake us.
        seal_lead = (carrying > 0 and moves_left <= 65
                     and score + carrying > max(0, their_load)
                     and (score <= their_load or moves_left <= dist_home + 12))
        go_home = carrying > 0 and (
            seal_lead or len(food) <= 2
            or carrying >= carry_limit
            or (0 < power_window <= dist_home + 5 and carrying >= 4)
            or (ghost_dist <= 4 and (carrying >= 3 or ghost_dist <= 2))
            or dist_home + 5 >= moves_left)

        if go_home:
            targets = self.home_border
            if capsules and ghost_dist <= 5:
                escape = min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
                if (self.get_maze_distance(my_pos, escape) <= 3
                        and self.get_maze_distance(my_pos, escape) < dist_home
                        and all(self.get_maze_distance(escape, g) >= 2 for g in ghosts)):
                    targets = [escape]
        elif ghost_dist <= 5 and capsules:
            # Chased with a capsule available: eating it scares the ghosts.
            targets = capsules
        else:
            targets = food
            if capsules and food and power_window < 5 and guard_timers:
                nearest_capsule = min(self.get_maze_distance(my_pos, c) for c in capsules)
                nearest_food = min(self.get_maze_distance(my_pos, f) for f in food)
                if nearest_capsule <= 8 and nearest_capsule <= nearest_food + 5:
                    targets = capsules

        action = self.first_step_towards(my_pos, targets, blocked)
        if action is None and targets is capsules:
            action = self.first_step_towards(my_pos, food, blocked)
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
                return action if action is not None else self.safe_fallback(
                    game_state, invaders, stay_home=True)
            intercept = self.pursue(game_state, my_pos, invaders)
            if intercept is not None:
                return intercept
            targets = invaders
        elif self.last_eaten is not None:
            targets = [self.last_eaten]
        else:
            targets = [self.choose_patrol_target(game_state, my_pos)]

        action = self.first_step_towards(my_pos, targets, self.enemy_side)
        if action is None:
            legal = []
            for candidate in game_state.get_legal_actions(self.index):
                if candidate == Directions.STOP:
                    continue
                dx, dy = Actions.direction_to_vector(candidate)
                next_pos = (int(my_pos[0] + dx), int(my_pos[1] + dy))
                if self.is_home(next_pos):
                    legal.append(candidate)
            action = random.choice(legal) if legal else Directions.STOP
        return action

    def pursue(self, state, position, invaders):
        """Plan a short interception with alternating ghost and invader moves."""
        enemies = [(q, agent.num_carrying) for _, agent, q in self.visible_enemies(state)
                   if agent.is_pacman and q in invaders]
        invader, load = min(enemies, key=lambda item: (
            self.get_maze_distance(position, item[0]) - min(6, item[1]) * 0.55))
        if load < 5 or self.get_maze_distance(position, invader) > 7:
            return None

        moves = {}
        homes = {}
        distances = {}
        def distance(a, b):
            key = (a, b) if a <= b else (b, a)
            if key not in distances:
                distances[key] = self.get_maze_distance(a, b)
            return distances[key]

        def home_distance(q):
            if q not in homes:
                homes[q] = self.distance_home(q)
            return homes[q]

        def adjacent(q):
            if q not in moves:
                moves[q] = self.neighbours(q) + [(q, Directions.STOP)]
            return moves[q]

        @lru_cache(maxsize=None)
        def value(guard, pacman, remaining, defensive_turn):
            if guard == pacman:
                return 100 + remaining
            if not self.is_home(pacman):
                return -100 - remaining
            if remaining == 0:
                gap = distance(guard, pacman)
                gates = [g for g in self.home_border if distance(g, pacman) <= home_distance(pacman) + 2]
                advantage = min(distance(g, pacman) - distance(g, guard) for g in gates)
                return -gap + min(4, max(-4, advantage)) * 1.1
            if defensive_turn:
                return max(value(g, pacman, remaining - 1, False)
                           for g, _ in adjacent(guard) if self.is_home(g))
            return min(value(guard, q, remaining - 1, True) for q, _ in adjacent(pacman))

        legal = state.get_legal_actions(self.index)
        preferred = self.first_step_towards(position, [invader], self.enemy_side)
        choices = [(q, action) for q, action in adjacent(position)
                   if self.is_home(q) and action in legal]
        if choices:
            return max(choices, key=lambda pair: (value(pair[0], invader, 5, False),
                                                  pair[1] == preferred, -distance(pair[0], invader)))[1]
        return None

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
