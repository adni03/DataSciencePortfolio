# TicTacToe AI

Fun little AI bot that plays TicTacToe.

<!--more-->

In this fun project, the goal was to apply my understanding of Machine Learning to build a TicTacToe player. The learning problem was defined based on Tom Mitchell’s example learning problem of a checkers learning algorithm in his textbook.

{{< admonition note "Setup" >}}
**Task T**: playing TicTacToe \
**Performance Measure P**: Percentage of games won against humans\
**Experience E**: Indirect feedback via solution trace generated from games played against itself
{{< /admonition >}}

### Learning from Experience
The target function (V) was chosen to be a linear function that maps a given board state to a real value (score). We then use an approximation algorithm `Least Mean Squares` to learn the target function from the solution trace.

{{< admonition note "Learning" >}}
 V(board_state) = R, (score for a given board state) \
 V_hat(board_state) = (w.T)*X (product of weights and corresponding feature values) \
 \
 The score (R) for each non-final board state isi assigned with the estimated score of the successor board state: \
    &emsp; V(board_state) = V_hat(successor(board_state)) \
    &emsp; V(final_board_state) = 100 (win) \
    &emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;0   (draw) \
    &emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;-100 (loss) 
{{< /admonition >}}

### Sample Game
```python
  print("Let the game begin : ")
  play(size)
  Let the game begin : 
  Computer's turn : 
  --------------------
  | - || - || - || X |
  --------------------
  | - || - || - || - |
  --------------------
  | - || - || - || - |
  --------------------
  | - || - || - || - |
  --------------------
  Human's move : 
  Enter x coordinate : 0
  Enter y coordinate : 0
  --------------------
  | O || - || - || X |
  --------------------
  | - || - || - || - |
  --------------------
  | - || - || - || - |
  --------------------
  | - || - || - || - |
  --------------------
  Computer's turn : 
  --------------------
  | O || - || - || X |
  --------------------
  | X || - || - || - |
  --------------------
  | - || - || - || - |
  --------------------
  | - || - || - || - |
  --------------------
  Human's move : 
  Enter x coordinate : 3
  Enter y coordinate : 1
  --------------------
  | O || - || - || X |
  --------------------
  | X || - || - || - |
  --------------------
  | - || - || - || - |
  --------------------
  | - || O || - || - |
  --------------------
  Computer's turn : 
  --------------------
  | O || - || - || X |
  --------------------
  | X || X || - || - |
  --------------------
  | - || - || - || - |
  --------------------
  | - || O || - || - |
  --------------------
  Human's move : 
  Enter x coordinate : 3
  Enter y coordinate : 2
  --------------------
  | O || - || - || X |
  --------------------
  | X || X || - || - |
  --------------------
  | - || - || - || - |
  --------------------
  | - || O || O || - |
  --------------------
  Computer's turn : 
  --------------------
  | O || - || - || X |
  --------------------
  | X || X || X || - |
  --------------------
  | - || - || - || - |
  --------------------
  | - || O || O || - |
  --------------------
  Human's move : 
  Enter x coordinate : 3
  Enter y coordinate : 3
  --------------------
  | O || - || - || X |
  --------------------
  | X || X || X || - |
  --------------------
  | - || - || - || - |
  --------------------
  | - || O || O || O |
  --------------------
  Computer's turn : 
  --------------------
  | O || - || - || X |
  --------------------
  | X || X || X || - |
  --------------------
  | - || - || - || - |
  --------------------
  | X || O || O || O |
  --------------------
  Human's move : 
  Enter x coordinate : 2
  Enter y coordinate : 1
  --------------------
  | O || - || - || X |
  --------------------
  | X || X || X || - |
  --------------------
  | - || O || - || - |
  --------------------
  | X || O || O || O |
  --------------------
  Computer's turn : 
  --------------------
  | O || - || - || X |
  --------------------
  | X || X || X || X |
  --------------------
  | - || O || - || - |
  --------------------
  | X || O || O || O |
  --------------------
  Computer Wins!
```

