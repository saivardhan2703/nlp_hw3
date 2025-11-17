# Q1) Char-RNN (O/P)

Vocab size: 68
Epoch 1, Loss = 2.4290
Epoch 2, Loss = 1.5520
Epoch 3, Loss = 0.9472
Epoch 4, Loss = 0.5330
Epoch 5, Loss = 0.3176

Sample (temp=0.7):
Holmes. “It is not an English
paper at all. Hold it up to the heling a
frome how nowe have breach ot
mesperian boline
do by at alp for elvein and observen. Then he stown an a mas are gince Hollover of his own deliather
of the Triented up
of the sing upon the happer aghat acroms al quarters an corman

Sample (temp=1.0):
Hold thear in tramely for exarning tall me through matterused a sucht ad his own a Caspas-ftincoceited im to mise from were Holmessan which had been lyen pasion sho eis, and thare of Eurine
am anate colserough ot  uodd his conters apper am lod hes see. Ind edming of the secinety—ere’s of viiniver. Ae

Sample (temp=1.2):
Hetcones it as high post is out he.    II hand eredog. AAd expenight—it a witaok. Hen’ he. we har papind. My twore,
sa, I dedvertion which had been hear you dee indeetersa
peer, which makny from the faclo musted my ow.”

“I tinse was up, a paused if theresson whredref in a
German’s mounce himse was

## Reflection
Increasing **sequence length** allows the RNN to learn longer-range dependencies, but it increases training cost and makes vanishing gradients more likely. A larger **hidden size** gives the model more expressive power but can lead to slower training and overfitting if the dataset is small. The **temperature parameter τ** directly influences sampling creativity: τ < 1 produces stable but repetitive text, τ ≈ 1 creates balanced and natural text, while τ > 1 results in more diverse but often noisy or ungrammatical outputs. These effects match the behavior shown in lecture examples.


#  Q2) Mini Transformer Encoder  (O/P)

Vocabulary (tokens): ['Sherlock', 'Holmes', 'loved', 'logic.', 'Watson', 'recorded', 'the', 'adventures.', 'The', 'detective', 'examined', 'clues', 'carefully.', 'A', 'case', 'begins', 'with', 'a', 'call.', 'They', 'walked', 'down', 'Baker', 'Street.', 'mystery', 'often', 'hides', 'in', 'plain', 'sight.', 'He', 'lit', 'his', 'pipe', 'and', 'thought.', 'client', 'explained', 'strange', 'events.', 'asked', 'few', 'precise', 'questions.', 'solved', 'puzzle', 'together.', '<PAD>']
Max tokens in a sentence: 7

Input tokens (first sentence): ['Sherlock', 'Holmes', 'loved', 'logic.']
Embedded input shape: torch.Size([10, 7, 64])

Final contextual embeddings shape: torch.Size([10, 7, 64])

Final embeddings (sentence 0, token 0..min(5,T)):
 token 0 (Sherlock): mean=0.0000, std=1.0000
 token 1 (Holmes): mean=0.0000, std=1.0000
 token 2 (loved): mean=0.0000, std=1.0000
 token 3 (logic.): mean=-0.0000, std=1.0000
 token 4 (<PAD>): mean=0.0000, std=1.0000

Saved attention heatmap to q2_attention_heatmap.png

Attention matrix (first 8 tokens):
[[0.149 0.226 0.105 0.184 0.106 0.112 0.117]
 [0.141 0.123 0.245 0.145 0.127 0.112 0.107]
 [0.199 0.143 0.149 0.163 0.114 0.115 0.116]
 [0.127 0.135 0.181 0.172 0.134 0.125 0.126]
 [0.125 0.211 0.165 0.225 0.098 0.089 0.088]
 [0.122 0.212 0.16  0.222 0.1   0.092 0.091]
 [0.117 0.209 0.157 0.211 0.107 0.1   0.1  ]]

 ## Reflection
 Embedded input shape: [10, 7, 64]  
Final contextual embeddings shape: [10, 7, 64]  
Attention heatmap saved as q2_attention_heatmap.png  


# Q3) Attention Test (O/P)
Attention weights WITHOUT scaling (shape): torch.Size([1, 5, 5])
Attention weights WITH scaling (shape): torch.Size([1, 5, 5])

Weights WITHOUT scaling (first query row):
[0. 0. 1. 0. 0.]

Weights WITH scaling (first query row):
[0.     0.     0.9976 0.     0.0024]

Sum of weights (sanity): 1.0 1.0

Output shape WITHOUT scaling: torch.Size([1, 5, 64])
Output shape WITH scaling: torch.Size([1, 5, 64])

Sample output vector (first token) WITHOUT scaling (first 6 dims):
[0.4974 0.2685 1.4769 0.3548 1.6247 0.5934]

Sample output vector (first token) WITH scaling (first 6 dims):
[0.4967 0.2654 1.476  0.3526 1.6158 0.5929]

Scores range WITHOUT scaling: min, max = -335.6247863769531 242.21969604492188
Scores range WITH scaling:    min, max = -41.95309829711914 30.277462005615234
