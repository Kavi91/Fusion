from graphviz import Digraph

dot = Digraph(comment='FusionLIVO Network')

# Nodes
dot.node('A', 'RGB Input\n(batch, 2, 6, 184, 608)')
dot.node('B', 'LiDAR Input\n(batch, 2, 1, 64, 900)')
dot.node('C', 'DeepVO CNN\nMulti-scale Features')
dot.node('D', 'LoRCoNLO CNN\nMulti-scale Features')
dot.node('E', 'FPN (RGB)\n256@64x900')
dot.node('F', 'FPN (LiDAR)\n256@64x900')
dot.node('G', 'Concatenation\n512@64x900')
dot.node('H', 'Attention\n1@64x900')
dot.node('I', 'Fused Features\n512@64x900')
dot.node('J', 'Fusion Conv\n256@64x900')
dot.node('K', 'Flatten\n256*64*900')
dot.node('L', 'Bi-LSTM\n2048')
dot.node('M', 'Linear\n6-DoF Poses')

# Edges
dot.edges(['AC', 'BD', 'CE', 'DF', 'EG', 'FG', 'GH', 'HI', 'IJ', 'JK', 'KL', 'LM'])

# Render
dot.render('FusionLIVO_network', view=True, format='png')