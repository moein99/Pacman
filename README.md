# Pacman

A self-designed pacman agents compete in a capture the flag style game of Pac-Man.

## Explanation & My Approach

In this project i've designed feature based agents. At any point, agents decide where to go based on some features which they consider.

### Offensive Agent

Generally it moves to opponent field and tries to capture foods. There are some features which it considers to play better.

**Features**:
- **return_to_base**:   
  In this feature, we first calculate the shortest distance of the agent to closest entrace of team field, then we apply square root to make this function non-linear 
  the reason is that i want my agent to get back to base and deliver foods if it is close and to keep gathering foods if it is far away from base. 
  
- **successor_score**:  
  In this feature, we get the number of foods that are in the next position on the ground in order 
  to direct our attacking agent toward eating them as much as possible.
  
- **closest_food**:  
  This feature guides the agent to closest food.
  
- **closest_capsule**:  
  This feature guides the agent to capsule, which is way more important than foods.

- **is_impasse**:  
  This feature stops the agent from moving to positions which there is no food in them and lead the agent to impasses. 
  
- **daed_impasse**:  
  This one is similar to previous feature, in some cases there are foods in impasses but if the agent tries to get the foods, it wont be able to get out because the ghost probably will block the way. 


### Defensive Agent

In base case, when there is no invaders, this agent stays on the nearest food to team boarder.

**Features**:
- **on_defense**:  
Keeps agent in defensive side.

- **num_invaders**:  
This feature is needed when distance to invader is 1 and the agent should destory it.

- **dis_to_eaten_food**:  
Guides the agent to invader. 

- **distance_to_border_food**:  
Makes the ghost stay on nearest food to boarder when there is no invader.
