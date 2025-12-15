# Goal Is Path Which Knight Should Visit Exactly 1.
# Complexity Is O(8n^2)
# Backtracking Algorithm
## In Backtracking You Start In Left Depth One Then if Get Sol.
## Else This Branch Is DeadEnd Leave this One And Take Another One Unlit Get Sol
# In Backtracking There Are 8 New Place As Max.
# (i+2, j+1) or (i+1, j+2)
# (i-1, j+2) or (i-2, j+1)
# (i-2, j-1) or (i-1, j-2)
# (i+1, j-2) or (i+2, j-1)
# Constraint:- Can not move in  Place Already Visited
### Recursion Knight Will Move Until This Way Be DeadEnd ==> Should Return to Position Who Start From & Re Try Again Until Found Sol.

# Cultural Algorithm "There are Population And Belief"
## Population Is Group Of Individual All Try To Solve Puzzle In Own Way
## Belief Is Cultural Evaluation "Shared Pool Of Wisdom" that Build From Best Attempts Over Time

### Population there is trail error happens Whole Collection Of individual Each One try to Solve Problem They Evolve Form One Generation to Another Exploring New path and Try Out New Strategies
### Belief Space "Game Changer": Record Of Single Best Sol No Its Like Library Of Wisdom Its Stores Shared Knowledge About What Makes path Good Path learn From All Successful Individuals Across Generations
### Knowledge in Belief Space Is Influence New Generation Of Individuals then The Most Successfully Individuals From New Generation Get Accepted Back In Belief Space Updating The Collective Wisdom For Next Generation
# We Start By Creating Initial Population and  Empty Belief Space
# Then Gump into Loop Generating Evaluating and Adapting
## Generating New Generation ==> Evaluate Individuals Of New Generations ==> Compare Individual By that found in Belief Space "By Taking the Best From New Generation" ==> Update The Belief Space With Better Individuals ==> Repeat Until Termination "Conditions are Satisfied"
###